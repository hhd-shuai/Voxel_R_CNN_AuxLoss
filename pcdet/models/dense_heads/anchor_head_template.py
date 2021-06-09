import numpy as np
import torch
import torch.nn as nn

from ...utils import box_coder_utils, common_utils, loss_utils
from .target_assigner.anchor_generator import AnchorGenerator
from .target_assigner.atss_target_assigner import ATSSTargetAssigner
from .target_assigner.axis_aligned_target_assigner import AxisAlignedTargetAssigner


class AnchorHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training): # model_cfg: {'NAME': 'AnchorHeadSingle', 'CLASS_AGNOSTIC': False, 'USE_DIRECTION_CLASSIFIER': True, 'DIR_OFFSET': 0.78539, 'DIR_LIMIT_OFFSET': 0.0, 'NUM_DIR_BINS': 2, 'ANCHOR_GENERATOR_CONFIG': [{'class_name': 'Car', 'anchor_sizes': [[3.9, 1.6, 1.56]], 'anchor_rotations': [0, 1.57], 'anchor_bottom_heights': [-1.78], 'align_center': False, 'feature_map_stride': 8, 'matched_threshold': 0.6, 'unmatched_threshold': 0.45}], 'TARGET_ASSIGNER_CONFIG': {'NAME': 'AxisAlignedTargetAssigner', 'POS_FRACTION': -1.0, 'SAMPLE_SIZE': 512, 'NORM_BY_NUM_EXAMPLES': False, 'MATCH_HEIGHT': False, 'BOX_CODER': 'ResidualCoder'}, 'LOSS_CONFIG': {'LOSS_WEIGHTS': {'cls_weight': 1.0, 'loc_weight': 2.0, 'dir_weight': 0.2, 'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}}}
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class# 1
        self.class_names = class_names #['Car']
        self.predict_boxes_when_training = predict_boxes_when_training #{'NAME': 'VoxelRCNNHead', 'CLASS_AGNOSTIC': True, 'SHARED_FC': [256, 256], 'CLS_FC': [256, 256], 'REG_FC': [256, 256], 'DP_RATIO': 0.3, 'NMS_CONFIG': {'TRAIN': {'NMS_TYPE': 'nms_gpu', 'MULTI_CLASSES_NMS': False, 'NMS_PRE_MAXSIZE': 9000, 'NMS_POST_MAXSIZE': 512, 'NMS_THRESH': 0.8}, 'TEST': {'NMS_TYPE': 'nms_gpu', 'MULTI_CLASSES_NMS': False, 'USE_FAST_NMS': True, 'SCORE_THRESH': 0.0, 'NMS_PRE_MAXSIZE': 2048, 'NMS_POST_MAXSIZE': 100, 'NMS_THRESH': 0.7}}, 'ROI_GRID_POOL': {'FEATURES_SOURCE': ['x_conv3', 'x_conv4'], 'PRE_MLP': True, 'GRID_SIZE': 6, 'POOL_LAYERS': {'x_conv3': {'MLPS': [[32, 32], [32, 32]], 'QUERY_RANGES': [[2, 2, 2], [4, 4, 4]], 'POOL_RADIUS': [0.4, 0.8], 'NSAMPLE': [16, 16], 'POOL_METHOD': 'max_pool'}, 'x_conv4': {'MLPS': [[32, 32], [32, 32]], 'QUERY_RANGES': [[2, 2, 2], [4, 4, 4]], 'POOL_RADIUS': [0.8, 1.6], 'NSAMPLE': [16, 16], 'POOL_METHOD': 'max_pool'}}}, 'TARGET_CONFIG': {'BOX_CODER': 'ResidualCoder', 'ROI_PER_IMAGE': 128, 'FG_RATIO': 0.5, 'SAMPLE_ROI_BY_EACH_CLASS': True, 'CLS_SCORE_TYPE': 'roi_iou', 'CLS_FG_THRESH': 0.75, 'CLS_BG_THRESH': 0.25, 'CLS_BG_THRESH_LO': 0.1, 'HARD_BG_RATIO': 0.8, 'REG_FG_THRESH': 0.55}, 'LOSS_CONFIG': {'CLS_LOSS': 'BinaryCrossEntropy', 'REG_LOSS': 'smooth-l1', 'CORNER_LOSS_REGULARIZATION': True, 'GRID_3D_IOU_LOSS': False, 'LOSS_WEIGHTS': {'rcnn_cls_weight': 1.0, 'rcnn_reg_weight': 1.0, 'rcnn_corner_weight': 1.0, 'rcnn_iou3d_weight': 1.0, 'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}}}
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False) #False

        anchor_target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG #anchor_target_cfg: {'NAME': 'AxisAlignedTargetAssigner', 'POS_FRACTION': -1.0, 'SAMPLE_SIZE': 512, 'NORM_BY_NUM_EXAMPLES': False, 'MATCH_HEIGHT': False, 'BOX_CODER': 'ResidualCoder'}
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)(
            num_dir_bins=anchor_target_cfg.get('NUM_DIR_BINS', 6),
            **anchor_target_cfg.get('BOX_CODER_CONFIG', {})
        )

        anchor_generator_cfg = self.model_cfg.ANCHOR_GENERATOR_CONFIG #[{'class_name': 'Car', 'anchor_sizes': [[3.9, 1.6, 1.56]], 'anchor_rotations': [0, 1.57], 'anchor_bottom_heights': [-1.78], 'align_center': False, 'feature_map_stride': 8, 'matched_threshold': 0.6, 'unmatched_threshold': 0.45}]
        anchors, self.num_anchors_per_location = self.generate_anchors(
            anchor_generator_cfg, grid_size=grid_size, point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size
        ) # num_anchors_per_location: [2]
        self.anchors = [x.cuda() for x in anchors]
        self.target_assigner = self.get_target_assigner(anchor_target_cfg)

        self.forward_ret_dict = {}
        self.build_losses(self.model_cfg.LOSS_CONFIG)

    @staticmethod
    def generate_anchors(anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7):
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg
        )
        feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in anchor_generator_cfg]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)

        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors

        return anchors_list, num_anchors_per_location_list

    def get_target_assigner(self, anchor_target_cfg):
        if anchor_target_cfg.NAME == 'ATSS':
            target_assigner = ATSSTargetAssigner(
                topk=anchor_target_cfg.TOPK,
                box_coder=self.box_coder,
                use_multihead=self.use_multihead,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        elif anchor_target_cfg.NAME == 'AxisAlignedTargetAssigner':
            target_assigner = AxisAlignedTargetAssigner(
                model_cfg=self.model_cfg,
                class_names=self.class_names,
                box_coder=self.box_coder,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        else:
            raise NotImplementedError
        return target_assigner

    def build_losses(self, losses_cfg): #{'LOSS_WEIGHTS': {'cls_weight': 1.0, 'loc_weight': 2.0, 'dir_weight': 0.2, 'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}}
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE #'WeightedSmoothL1Loss'
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8) torch.Size([8, 15, 8])
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets(
            self.anchors, gt_boxes #self.anchors: torch.Size([1, 200, 176, 1, 2, 7]) gt_boxes: (B, M, 8) torch.Size([8, 15, 8])
        ) #'box_cls_labels':torch.Size([8, 70400])  'box_reg_targets':torch.Size([8, 70400, 7])  'reg_weights':torch.Size([8, 70400])
        return targets_dict

    def get_cls_layer_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds'] #torch.Size([8, 200, 176, 2])
        box_cls_labels = self.forward_ret_dict['box_cls_labels'] #torch.Size([8, 70400])
        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]  torch.Size([8, 70400])
        positives = box_cls_labels > 0 #torch.Size([8, 70400])
        negatives = box_cls_labels == 0 #torch.Size([8, 70400])
        negative_cls_weights = negatives * 1.0 #torch.Size([8, 70400])
        cls_weights = (negative_cls_weights + 1.0 * positives).float() #torch.Size([8, 70400])
        reg_weights = positives.float() #torch.Size([8, 70400])
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        pos_normalizer = positives.sum(1, keepdim=True).float() #torch.Size([8, 1])
        reg_weights /= torch.clamp(pos_normalizer, min=1.0) #torch.Size([8, 70400])
        cls_weights /= torch.clamp(pos_normalizer, min=1.0) #torch.Size([8, 70400])
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels) #torch.Size([8, 70400])
        cls_targets = cls_targets.unsqueeze(dim=-1) #torch.Size([8, 70400, 1])

        cls_targets = cls_targets.squeeze(dim=-1) #torch.Size([8, 70400])
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        ) #torch.Size([8, 70400, 2])
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class) #torch.Size([8, 70400, 1])
        one_hot_targets = one_hot_targets[..., 1:] #torch.Size([8, 70400, 1])
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]  torch.Size([8, 70400, 1])
        cls_loss = cls_loss_src.sum() / batch_size #tensor(0.0518, device='cuda:0', grad_fn=<DivBackward0>)

        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2): # anchors： torch.Size([8, 70400, 7])  reg_targets：torch.Size([8, 70400, 7])  dir_offset： 0.78539  num_bins：2
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1]) #torch.Size([8, 70400, 7])
        rot_gt = reg_targets[..., 6] + anchors[..., 6] #torch.Size([8, 70400])
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi) #torch.Size([8, 70400])
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()  #torch.Size([8, 70400])
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1) #torch.Size([8, 70400])

        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device) #torch.Size([8, 70400, 2])
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets #torch.Size([8, 70400, 2])
        return dir_cls_targets

    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds'] #torch.Size([8, 200, 176, 14])
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None) #torch.Size([8, 200, 176, 4])
        box_reg_targets = self.forward_ret_dict['box_reg_targets'] #torch.Size([8, 70400, 7])
        box_cls_labels = self.forward_ret_dict['box_cls_labels'] #torch.Size([8, 70400])
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0  #torch.Size([8, 70400])
        reg_weights = positives.float() #torch.Size([8, 70400])
        pos_normalizer = positives.sum(1, keepdim=True).float() #torch.Size([8, 1])
        reg_weights /= torch.clamp(pos_normalizer, min=1.0) #torch.Size([8, 70400])

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3) #torch.Size([1, 200, 176, 1, 2, 7])
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1) #torch.Size([8, 70400, 7])
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])
        # sin(a - b) = sinacosb-cosasinb     torch.Size([8, 70400, 7])
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets) #box_preds_sin: torch.Size([8, 70400, 7]), reg_targets_sin: torch.Size([8, 70400, 7])
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M] torch.Size([8, 70400, 7])
        loc_loss = loc_loss_src.sum() / batch_size #tensor(0.0577, device='cuda:0', grad_fn=<DivBackward0>)

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight'] #weight 2.0
        box_loss = loc_loss #tensor(0.1154, device='cuda:0', grad_fn=<MulBackward0>)
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            ) #torch.Size([8, 70400, 2])

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS) #torch.Size([8, 70400, 2])
            weights = positives.type_as(dir_logits) #torch.Size([8, 70400])
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0) #torch.Size([8, 70400])
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights) #torch.Size([8, 70400])
            dir_loss = dir_loss.sum() / batch_size #tensor(0.0275, device='cuda:0', grad_fn=<DivBackward0>)
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight'] #tensor(0.0055, device='cuda:0', grad_fn=<MulBackward0>)
            box_loss += dir_loss #tensor(0.1209, device='cuda:0', grad_fn=<AddBackward0>)
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss() #cls_loss： tensor(0.0518, device='cuda:0', grad_fn=<MulBackward0>)  tb_dict: {'rpn_loss_cls': 0.05177481472492218}
        box_loss, tb_dict_box = self.get_box_reg_layer_loss() #tensor(0.1209, device='cuda:0', grad_fn=<AddBackward0>)  {'rpn_loss_loc': 0.11539990454912186, 'rpn_loss_dir': 0.0054987650364637375}
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss #tensor(0.1727, device='cuda:0', grad_fn=<AddBackward0>)

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:  8
            cls_preds: (N, H, W, C1)  torch.Size([8, 200, 176, 2])
            box_preds: (N, H, W, C2)  torch.Size([8, 200, 176, 14])
            dir_cls_preds: (N, H, W, C3)  torch.Size([8, 200, 176, 4])

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3) #torch.Size([1, 200, 176, 1, 2, 7])
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0] #70400
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1) #torch.Size([8, 70400, 7])
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds #torch.Size([8, 70400, 1])
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1) #torch.Size([8, 70400, 7])
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors) #torch.Size([8, 70400, 7])

        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET #0.78539
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET #0.0
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1) #torch.Size([8, 70400, 2])
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1] #torch.Size([8, 70400])

            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS) #3.141592653589793
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            ) #torch.Size([8, 70400])
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype) #torch.Size([8, 70400, 7])

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )

        return batch_cls_preds, batch_box_preds #batch_cls_preds: torch.Size([8, 70400, 1])  batch_box_preds: torch.Size([8, 70400, 7])

    def forward(self, **kwargs):
        raise NotImplementedError
