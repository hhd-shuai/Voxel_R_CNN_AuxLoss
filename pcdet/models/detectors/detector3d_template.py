import os

import torch
import torch.nn as nn

from ...ops.iou3d_nms import iou3d_nms_utils
from .. import backbones_2d, backbones_3d, dense_heads, roi_heads
from ..backbones_2d import map_to_bev
from ..backbones_3d import pfe, vfe
from ..model_utils import model_nms_utils


class Detector3DTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, dataset): # self:VoxelRCNN(),  model_cfg: {'NAME': 'VoxelRCNN', 'VFE': {'NAME': 'MeanVFE'}, 'BACKBONE_3D': {'NAME': 'VoxelBackBone8x'}, 'MAP_TO_BEV': {'NAME': 'HeightCompression', 'NUM_BEV_FEATURES': 256}, 'BACKBONE_2D': {'NAME': 'BaseBEVBackbone', 'LAYER_NUMS': [4, 4], 'LAYER_STRIDES': [1, 2], 'NUM_FILTERS': [64, 128], 'UPSAMPLE_STRIDES': [1, 2], 'NUM_UPSAMPLE_FILTERS': [128, 128]}, 'DENSE_HEAD': {'NAME': 'AnchorHeadSingle', 'CLASS_AGNOSTIC': False, 'USE_DIRECTION_CLASSIFIER': True, 'DIR_OFFSET': 0.78539, 'DIR_LIMIT_OFFSET': 0.0, 'NUM_DIR_BINS': 2, 'ANCHOR_GENERATOR_CONFIG': [{'class_name': 'Car', 'anchor_sizes': [[3.9, 1.6, 1.56]], 'anchor_rotations': [0, 1.57], 'anchor_bottom_heights': [-1.78], 'align_center': False, 'feature_map_stride': 8, 'matched_threshold': 0.6, 'unmatched_threshold': 0.45}], 'TARGET_ASSIGNER_CONFIG': {'NAME': 'AxisAlignedTargetAssigner', 'POS_FRACTION': -1.0, 'SAMPLE_SIZE': 512, 'NORM_BY_NUM_EXAMPLES': False, 'MATCH_HEIGHT': False, 'BOX_CODER': 'ResidualCoder'}, 'LOSS_CONFIG': {'LOSS_WEIGHTS': {'cls_weight': 1.0, 'loc_weight': 2.0, 'dir_weight': 0.2, 'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}}}, 'ROI_HEAD': {'NAME': 'VoxelRCNNHead', 'CLASS_AGNOSTIC': True, 'SHARED_FC': [256, 256], 'CLS_FC': [256, 256], 'REG_FC': [256, 256], 'DP_RATIO': 0.3, 'NMS_CONFIG': {'TRAIN': {'NMS_TYPE': 'nms_gpu', 'MULTI_CLASSES_NMS': False, 'NMS_PRE_MAXSIZE': 9000, 'NMS_POST_MAXSIZE': 512, 'NMS_THRESH': 0.8}, 'TEST': {'NMS_TYPE': 'nms_gpu', 'MULTI_CLASSES_NMS': False, 'USE_FAST_NMS': True, 'SCORE_THRESH': 0.0, 'NMS_PRE_MAXSIZE': 2048, 'NMS_POST_MAXSIZE': 100, 'NMS_THRESH': 0.7}}, 'ROI_GRID_POOL': {'FEATURES_SOURCE': ['x_conv3', 'x_conv4'], 'PRE_MLP': True, 'GRID_SIZE': 6, 'POOL_LAYERS': {'x_conv3': {'MLPS': [[32, 32], [32, 32]], 'QUERY_RANGES': [[2, 2, 2], [4, 4, 4]], 'POOL_RADIUS': [0.4, 0.8], 'NSAMPLE': [16, 16], 'POOL_METHOD': 'max_pool'}, 'x_conv4': {'MLPS': [[32, 32], [32, 32]], 'QUERY_RANGES': [[2, 2, 2], [4, 4, 4]], 'POOL_RADIUS': [0.8, 1.6], 'NSAMPLE': [16, 16], 'POOL_METHOD': 'max_pool'}}}, 'TARGET_CONFIG': {'BOX_CODER': 'ResidualCoder', 'ROI_PER_IMAGE': 128, 'FG_RATIO': 0.5, 'SAMPLE_ROI_BY_EACH_CLASS': True, 'CLS_SCORE_TYPE': 'roi_iou', 'CLS_FG_THRESH': 0.75, 'CLS_BG_THRESH': 0.25, 'CLS_BG_THRESH_LO': 0.1, 'HARD_BG_RATIO': 0.8, 'REG_FG_THRESH': 0.55}, 'LOSS_CONFIG': {'CLS_LOSS': 'BinaryCrossEntropy', 'REG_LOSS': 'smooth-l1', 'CORNER_LOSS_REGULARIZATION': True, 'GRID_3D_IOU_LOSS': False, 'LOSS_WEIGHTS': {'rcnn_cls_weight': 1.0, 'rcnn_reg_weight': 1.0, 'rcnn_corner_weight': 1.0, 'rcnn_iou3d_weight': 1.0, 'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}}}, 'POST_PROCESSING': {'RECALL_THRESH_LIST': [0.3, 0.5, 0.7], 'SCORE_THRESH': 0.3, 'OUTPUT_RAW_SCORE': False, 'EVAL_METRIC': 'kitti', 'NMS_CONFIG': {'MULTI_CLASSES_NMS': False, 'NMS_TYPE': 'nms_gpu', 'NMS_THRESH': 0.1, 'NMS_PRE_MAXSIZE': 4096, 'NMS_POST_MAXSIZE': 500}}}
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = dataset.class_names #class_names: ['Car']
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'point_head', 'dense_head', 'roi_head'
        ]

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self): # VoxelRCNN()
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features, # 4
            'num_point_features': self.dataset.point_feature_encoder.num_point_features, # 4
            'grid_size': self.dataset.grid_size, # [1408 1600   40]
            'point_cloud_range': self.dataset.point_cloud_range, # [  0.  -40.   -3.   70.4  40.    1. ]
            'voxel_size': self.dataset.voxel_size # [0.05, 0.05, 0.1]
        }
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def build_vfe(self, model_info_dict):
        if self.model_cfg.get('VFE', None) is None:
            return None, model_info_dict

        vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](
            model_cfg=self.model_cfg.VFE,
            num_point_features=model_info_dict['num_rawpoint_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size']
        )
        model_info_dict['num_point_features'] = vfe_module.get_output_feature_dim()
        model_info_dict['module_list'].append(vfe_module)
        return vfe_module, model_info_dict

    def build_backbone_3d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_3D', None) is None:
            return None, model_info_dict

        backbone_3d_module = backbones_3d.__all__[self.model_cfg.BACKBONE_3D.NAME]( #  self.model_cfg.BACKBONE_3D.NAME: 
            model_cfg=self.model_cfg.BACKBONE_3D, #{'NAME': ''}
            input_channels=model_info_dict['num_point_features'], #4
            grid_size=model_info_dict['grid_size'], #[1408 1600   40]
            voxel_size=model_info_dict['voxel_size'], #[0.05, 0.05, 0.1]
            point_cloud_range=model_info_dict['point_cloud_range'] #[  0.  -40.   -3.   70.4  40.    1. ]
        )
        model_info_dict['module_list'].append(backbone_3d_module) # MeanVFE(), VoxelBackBone8x
        model_info_dict['num_point_features'] = backbone_3d_module.num_point_features #{'x_conv1': 16, 'x_conv2': 32, 'x_conv3': 64, 'x_conv4': 64}
        return backbone_3d_module, model_info_dict

    def build_map_to_bev_module(self, model_info_dict):
        if self.model_cfg.get('MAP_TO_BEV', None) is None:
            return None, model_info_dict

        map_to_bev_module = map_to_bev.__all__[self.model_cfg.MAP_TO_BEV.NAME](  #__all__ = {'HeightCompression': HeightCompression,'PointPillarScatter': PointPillarScatter} ,,,, 'HeightCompression'
            model_cfg=self.model_cfg.MAP_TO_BEV, #{'NAME': 'HeightCompression', 'NUM_BEV_FEATURES': 256}
            grid_size=model_info_dict['grid_size'] #[1408 1600   40]
        )
        model_info_dict['module_list'].append(map_to_bev_module)  #MeanVFE(), VoxelBackBone8x ,HeightCompression()
        model_info_dict['num_bev_features'] = map_to_bev_module.num_bev_features # 256
        return map_to_bev_module, model_info_dict

    def build_backbone_2d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_2D', None) is None:
            return None, model_info_dict

        backbone_2d_module = backbones_2d.__all__[self.model_cfg.BACKBONE_2D.NAME]( # 'BaseBEVBackbone'
            model_cfg=self.model_cfg.BACKBONE_2D, #{'NAME': 'BaseBEVBackbone', 'LAYER_NUMS': [4, 4], 'LAYER_STRIDES': [1, 2], 'NUM_FILTERS': [64, 128], 'UPSAMPLE_STRIDES': [1, 2], 'NUM_UPSAMPLE_FILTERS': [128, 128]}
            input_channels=model_info_dict['num_bev_features'] #256
        )
        model_info_dict['module_list'].append(backbone_2d_module) # MeanVFE(), VoxelBackBone8x, HeightCompression(), BaseBEVBackbone
        model_info_dict['num_bev_features'] = backbone_2d_module.num_bev_features #256
        return backbone_2d_module, model_info_dict

    def build_pfe(self, model_info_dict):
        if self.model_cfg.get('PFE', None) is None:
            return None, model_info_dict

        pfe_module = pfe.__all__[self.model_cfg.PFE.NAME](
            model_cfg=self.model_cfg.PFE,
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            num_bev_features=model_info_dict['num_bev_features'],
            num_rawpoint_features=model_info_dict['num_rawpoint_features']
        )
        model_info_dict['module_list'].append(pfe_module)
        model_info_dict['num_point_features'] = pfe_module.num_point_features
        model_info_dict['num_point_features_before_fusion'] = pfe_module.num_point_features_before_fusion
        return pfe_module, model_info_dict

    def build_dense_head(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD', None) is None:
            return None, model_info_dict
        dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD.NAME]( #'AnchorHeadSingle'
            model_cfg=self.model_cfg.DENSE_HEAD, #{'NAME': 'AnchorHeadSingle', 'CLASS_AGNOSTIC': False, 'USE_DIRECTION_CLASSIFIER': True, 'DIR_OFFSET': 0.78539, 'DIR_LIMIT_OFFSET': 0.0, 'NUM_DIR_BINS': 2, 'ANCHOR_GENERATOR_CONFIG': [{'class_name': 'Car', 'anchor_sizes': [[3.9, 1.6, 1.56]], 'anchor_rotations': [0, 1.57], 'anchor_bottom_heights': [-1.78], 'align_center': False, 'feature_map_stride': 8, 'matched_threshold': 0.6, 'unmatched_threshold': 0.45}], 'TARGET_ASSIGNER_CONFIG': {'NAME': 'AxisAlignedTargetAssigner', 'POS_FRACTION': -1.0, 'SAMPLE_SIZE': 512, 'NORM_BY_NUM_EXAMPLES': False, 'MATCH_HEIGHT': False, 'BOX_CODER': 'ResidualCoder'}, 'LOSS_CONFIG': {'LOSS_WEIGHTS': {'cls_weight': 1.0, 'loc_weight': 2.0, 'dir_weight': 0.2, 'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}}}
            input_channels=model_info_dict['num_bev_features'], #256
            num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1, #1
            class_names=self.class_names, #['Car']
            grid_size=model_info_dict['grid_size'], #[1408 1600   40]
            point_cloud_range=model_info_dict['point_cloud_range'], #[  0.  -40.   -3.   70.4  40.    1. ]
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False) #{'NAME': 'VoxelRCNNHead', 'CLASS_AGNOSTIC': True, 'SHARED_FC': [256, 256], 'CLS_FC': [256, 256], 'REG_FC': [256, 256], 'DP_RATIO': 0.3, 'NMS_CONFIG': {'TRAIN': {'NMS_TYPE': 'nms_gpu', 'MULTI_CLASSES_NMS': False, 'NMS_PRE_MAXSIZE': 9000, 'NMS_POST_MAXSIZE': 512, 'NMS_THRESH': 0.8}, 'TEST': {'NMS_TYPE': 'nms_gpu', 'MULTI_CLASSES_NMS': False, 'USE_FAST_NMS': True, 'SCORE_THRESH': 0.0, 'NMS_PRE_MAXSIZE': 2048, 'NMS_POST_MAXSIZE': 100, 'NMS_THRESH': 0.7}}, 'ROI_GRID_POOL': {'FEATURES_SOURCE': ['x_conv3', 'x_conv4'], 'PRE_MLP': True, 'GRID_SIZE': 6, 'POOL_LAYERS': {'x_conv3': {'MLPS': [[32, 32], [32, 32]], 'QUERY_RANGES': [[2, 2, 2], [4, 4, 4]], 'POOL_RADIUS': [0.4, 0.8], 'NSAMPLE': [16, 16], 'POOL_METHOD': 'max_pool'}, 'x_conv4': {'MLPS': [[32, 32], [32, 32]], 'QUERY_RANGES': [[2, 2, 2], [4, 4, 4]], 'POOL_RADIUS': [0.8, 1.6], 'NSAMPLE': [16, 16], 'POOL_METHOD': 'max_pool'}}}, 'TARGET_CONFIG': {'BOX_CODER': 'ResidualCoder', 'ROI_PER_IMAGE': 128, 'FG_RATIO': 0.5, 'SAMPLE_ROI_BY_EACH_CLASS': True, 'CLS_SCORE_TYPE': 'roi_iou', 'CLS_FG_THRESH': 0.75, 'CLS_BG_THRESH': 0.25, 'CLS_BG_THRESH_LO': 0.1, 'HARD_BG_RATIO': 0.8, 'REG_FG_THRESH': 0.55}, 'LOSS_CONFIG': {'CLS_LOSS': 'BinaryCrossEntropy', 'REG_LOSS': 'smooth-l1', 'CORNER_LOSS_REGULARIZATION': True, 'GRID_3D_IOU_LOSS': False, 'LOSS_WEIGHTS': {'rcnn_cls_weight': 1.0, 'rcnn_reg_weight': 1.0, 'rcnn_corner_weight': 1.0, 'rcnn_iou3d_weight': 1.0, 'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}}}
        )
        model_info_dict['module_list'].append(dense_head_module) # MeanVFE(), VoxelBackBone8x, HeightCompression(), BaseBEVBackbone
        return dense_head_module, model_info_dict

    def build_point_head(self, model_info_dict):
        if self.model_cfg.get('POINT_HEAD', None) is None:
            return None, model_info_dict

        point_head_module = dense_heads.__all__[self.model_cfg.POINT_HEAD.NAME](
            model_cfg=self.model_cfg.POINT_HEAD,
            num_class=self.num_class if not self.model_cfg.POINT_HEAD.CLASS_AGNOSTIC else 1,
        )

        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def build_roi_head(self, model_info_dict):
        if self.model_cfg.get('ROI_HEAD', None) is None:
            return None, model_info_dict
        point_head_module = roi_heads.__all__[self.model_cfg.ROI_HEAD.NAME](
            model_cfg=self.model_cfg.ROI_HEAD,
            input_channels=model_info_dict['num_point_features'], #{'x_conv1': 16, 'x_conv2': 32, 'x_conv3': 64, 'x_conv4': 64}
            point_cloud_range=model_info_dict['point_cloud_range'], #[  0.  -40.   -3.   70.4  40.    1. ]
            voxel_size=model_info_dict['voxel_size'],
            num_class=self.num_class if not self.model_cfg.ROI_HEAD.CLASS_AGNOSTIC else 1,
        )

        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def forward(self, **kwargs):
        raise NotImplementedError

    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds

            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]

                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict

    @staticmethod
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict

        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])

            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in self.state_dict() and self.state_dict()[key].shape == model_state_disk[key].shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self.load_state_dict(checkpoint['model_state'])

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch
