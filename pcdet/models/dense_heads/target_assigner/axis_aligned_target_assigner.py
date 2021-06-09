import numpy as np
import torch

from ....ops.iou3d_nms import iou3d_nms_utils
from ....utils import box_utils


class AxisAlignedTargetAssigner(object):
    def __init__(self, model_cfg, class_names, box_coder, match_height=False):
        super().__init__()

        anchor_generator_cfg = model_cfg.ANCHOR_GENERATOR_CONFIG
        anchor_target_cfg = model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = box_coder
        self.match_height = match_height
        self.class_names = np.array(class_names)
        self.anchor_class_names = [config['class_name'] for config in anchor_generator_cfg]
        self.pos_fraction = anchor_target_cfg.POS_FRACTION if anchor_target_cfg.POS_FRACTION >= 0 else None
        self.sample_size = anchor_target_cfg.SAMPLE_SIZE
        self.norm_by_num_examples = anchor_target_cfg.NORM_BY_NUM_EXAMPLES
        self.matched_thresholds = {}
        self.unmatched_thresholds = {}
        for config in anchor_generator_cfg:
            self.matched_thresholds[config['class_name']] = config['matched_threshold']
            self.unmatched_thresholds[config['class_name']] = config['unmatched_threshold']
         
        self.use_multihead = model_cfg.get('USE_MULTIHEAD', False)
        self.seperate_multihead = model_cfg.get('SEPERATE_MULTIHEAD', False)
        if self.seperate_multihead:
            rpn_head_cfgs = model_cfg.RPN_HEAD_CFGS
            self.gt_remapping = {}
            for rpn_head_cfg in rpn_head_cfgs:
                for idx, name in enumerate(rpn_head_cfg['HEAD_CLS_NAME']):
                    self.gt_remapping[name] = idx + 1

    def assign_targets(self, all_anchors, gt_boxes_with_classes):
        """
        Args:
            all_anchors: [(N, 7), ...] torch.Size([1, 200, 176, 1, 2, 7])
            gt_boxes_with_classes: (B, M, 8) torch.Size([8, 15, 8])
        Returns:

        """

        bbox_targets = []
        cls_labels = []
        reg_weights = []

        batch_size = gt_boxes_with_classes.shape[0] #8
        gt_classes = gt_boxes_with_classes[:, :, -1] #torch.Size([8, 15])
        gt_boxes = gt_boxes_with_classes[:, :, :-1] #torch.Size([8, 15, 7])
        for k in range(batch_size):
            cur_gt = gt_boxes[k] # k=0:torch.Size([15, 7])
            cnt = cur_gt.__len__() - 1 #10
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1] #torch.Size([11, 7])
            cur_gt_classes = gt_classes[k][:cnt + 1].int() # 11 tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], device='cuda:0', dtype=torch.int32)

            target_list = []
            for anchor_class_name, anchors in zip(self.anchor_class_names, all_anchors): #self.anchor_class_names: car
                if cur_gt_classes.shape[0] > 1:
                    mask = torch.from_numpy(self.class_names[cur_gt_classes.cpu() - 1] == anchor_class_name) #tensor([True, True, True, True, True, True, True, True, True])
                else:
                    mask = torch.tensor([self.class_names[c - 1] == anchor_class_name
                                         for c in cur_gt_classes], dtype=torch.bool)

                if self.use_multihead:
                    anchors = anchors.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchors.shape[-1])
                    if self.seperate_multihead:
                        selected_classes = cur_gt_classes[mask].clone()
                        if len(selected_classes) > 0:
                            new_cls_id = self.gt_remapping[anchor_class_name]
                            selected_classes[:] = new_cls_id
                    else:
                        selected_classes = cur_gt_classes[mask]
                else:
                    feature_map_size = anchors.shape[:3] #torch.Size([1, 200, 176])
                    anchors = anchors.view(-1, anchors.shape[-1]) #torch.Size([70400, 7])
                    selected_classes = cur_gt_classes[mask]  #tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], device='cuda:0', dtype=torch.int32)

                single_target = self.assign_targets_single(
                    anchors,
                    cur_gt[mask],
                    gt_classes=selected_classes,
                    matched_threshold=self.matched_thresholds[anchor_class_name],
                    unmatched_threshold=self.unmatched_thresholds[anchor_class_name]
                ) # dict:3  ,,'box_cls_labels':70400    'box_reg_targets':torch.Size([70400, 7])    'reg_weights':70400
                target_list.append(single_target)

            if self.use_multihead:
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(-1) for t in target_list],
                    'box_reg_targets': [t['box_reg_targets'].view(-1, self.box_coder.code_size) for t in target_list],
                    'reg_weights': [t['reg_weights'].view(-1) for t in target_list]
                }

                target_dict['box_reg_targets'] = torch.cat(target_dict['box_reg_targets'], dim=0)
                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=0).view(-1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=0).view(-1)
            else:
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(*feature_map_size, -1) for t in target_list],
                    'box_reg_targets': [t['box_reg_targets'].view(*feature_map_size, -1, self.box_coder.code_size)
                                        for t in target_list],
                    'reg_weights': [t['reg_weights'].view(*feature_map_size, -1) for t in target_list]
                } #'box_cls_labels'list[0]:torch.Size([1, 200, 176, 2])   torch.Size([1, 200, 176, 2, 7])   torch.Size([1, 200, 176, 2])
                target_dict['box_reg_targets'] = torch.cat(
                    target_dict['box_reg_targets'], dim=-2
                ).view(-1, self.box_coder.code_size) #torch.Size([70400, 7])

                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=-1).view(-1) #70400
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=-1).view(-1) #70400

            bbox_targets.append(target_dict['box_reg_targets'])
            cls_labels.append(target_dict['box_cls_labels'])
            reg_weights.append(target_dict['reg_weights'])

        bbox_targets = torch.stack(bbox_targets, dim=0) #torch.Size([8, 70400, 7])

        cls_labels = torch.stack(cls_labels, dim=0) #torch.Size([8, 70400])
        reg_weights = torch.stack(reg_weights, dim=0) #torch.Size([8, 70400])
        all_targets_dict = {
            'box_cls_labels': cls_labels,
            'box_reg_targets': bbox_targets,
            'reg_weights': reg_weights

        }
        return all_targets_dict

    def assign_targets_single(self, anchors, #torch.Size([70400, 7])
                         gt_boxes, #torch.Size([9, 7])
                         gt_classes, #9
                         matched_threshold=0.6,
                         unmatched_threshold=0.45
                        ):

        num_anchors = anchors.shape[0] #70400
        num_gt = gt_boxes.shape[0] # 11

        labels = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1 #70400
        gt_ids = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1 #70400

        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(anchors[:, 0:7], gt_boxes[:, 0:7]) \
                if self.match_height else box_utils.boxes3d_nearest_bev_iou(anchors[:, 0:7], gt_boxes[:, 0:7]) #torch.Size([70400, 9])

            anchor_to_gt_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=1)).cuda() #70400
            anchor_to_gt_max = anchor_by_gt_overlap[
                torch.arange(num_anchors, device=anchors.device), anchor_to_gt_argmax #70400  70400
            ] #70400

            gt_to_anchor_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=0)).cuda() #tensor([47380, 46704, 46036, 31401, 31064, 32530, 33832, 45908, 37101],device='cuda:0')
            gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, torch.arange(num_gt, device=anchors.device)] #tensor([0.8718, 0.8382, 0.7317, 0.8331, 0.6950, 0.6987, 0.8262, 0.7568, 0.8157],device='cuda:0')
            empty_gt_mask = gt_to_anchor_max == 0 #tensor([False, False, False, False, False, False, False, False, False],device='cuda:0')
            gt_to_anchor_max[empty_gt_mask] = -1

            anchors_with_max_overlap = (anchor_by_gt_overlap == gt_to_anchor_max).nonzero()[:, 0] #tensor([31064, 31401, 32530, 32532, 33832, 37101, 45908, 46036, 46704, 47380],device='cuda:0')
            gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap] #tensor([4, 3, 5, 5, 6, 8, 7, 2, 1, 0], device='cuda:0')
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force] #70400
            gt_ids[anchors_with_max_overlap] = gt_inds_force.int() #70400

            pos_inds = anchor_to_gt_max >= matched_threshold # matched_threshold： 0.6 ，，， 70400 tensor([False, False, False,  ..., False, False, False], device='cuda:0')
            gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds] #tensor([4, 4, 4, 3, 4, 4, 4, 3, 3, 3, 3, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8, 2, 2, 2, 7, 7, 7, 2, 2, 2, 7, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0],device='cuda:0')
            labels[pos_inds] = gt_classes[gt_inds_over_thresh] #70400
            gt_ids[pos_inds] = gt_inds_over_thresh.int() #70400
            bg_inds = (anchor_to_gt_max < unmatched_threshold).nonzero()[:, 0] # unmatched_threshold： 0.45 ，，，70292
        else:
            bg_inds = torch.arange(num_anchors, device=anchors.device)

        fg_inds = (labels > 0).nonzero()[:, 0] #tensor([30710, 30712, 30714, 31049, 31062, 31064, 31066, 31401, 31403, 31753, 32105, 32178, 32180, 32530, 32532, 32534, 33828, 33830, 33832, 33834, 34184, 36749, 37101, 37103, 37453, 37455, 37805, 45682, 45684, 45686, 45906, 45908, 45910, 46034, 46036, 46038, 46260, 46700, 46702, 46704, 46706, 47028, 47054, 47056, 47378, 47380, 47382, 47384],device='cuda:0')

        if self.pos_fraction is not None:
            num_fg = int(self.pos_fraction * self.sample_size)
            if len(fg_inds) > num_fg:
                num_disabled = len(fg_inds) - num_fg
                disable_inds = torch.randperm(len(fg_inds))[:num_disabled]
                labels[disable_inds] = -1
                fg_inds = (labels > 0).nonzero()[:, 0]

            num_bg = self.sample_size - (labels > 0).sum()
            if len(bg_inds) > num_bg:
                enable_inds = bg_inds[torch.randint(0, len(bg_inds), size=(num_bg,))]
                labels[enable_inds] = 0
            # bg_inds = torch.nonzero(labels == 0)[:, 0]
        else:
            if len(gt_boxes) == 0 or anchors.shape[0] == 0:
                labels[:] = 0
            else:
                labels[bg_inds] = 0
                labels[anchors_with_max_overlap] = gt_classes[gt_inds_force] #tensor([0, 0, 0,  ..., 0, 0, 0], device='cuda:0', dtype=torch.int32)

        bbox_targets = anchors.new_zeros((num_anchors, self.box_coder.code_size)) #torch.Size([70400, 7])
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            fg_gt_boxes = gt_boxes[anchor_to_gt_argmax[fg_inds], :] #torch.Size([48, 7])
            fg_anchors = anchors[fg_inds, :] #torch.Size([48, 7])
            bbox_targets[fg_inds, :] = self.box_coder.encode_torch(fg_gt_boxes, fg_anchors)

        reg_weights = anchors.new_zeros((num_anchors,))

        if self.norm_by_num_examples:
            num_examples = (labels >= 0).sum()
            num_examples = num_examples if num_examples > 1.0 else 1.0
            reg_weights[labels > 0] = 1.0 / num_examples
        else:
            reg_weights[labels > 0] = 1.0 #---

        ret_dict = {
            'box_cls_labels': labels, #70400
            'box_reg_targets': bbox_targets, #torch.Size([70400, 7])
            'reg_weights': reg_weights, #70400
        }
        return ret_dict
