import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate


class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        ) #{'NAME': 'AnchorHeadSingle', 'CLASS_AGNOSTIC': False, 'USE_DIRECTION_CLASSIFIER': True, 'DIR_OFFSET': 0.78539, 'DIR_LIMIT_OFFSET': 0.0, 'NUM_DIR_BINS': 2, 'ANCHOR_GENERATOR_CONFIG': [{'class_name': 'Car', 'anchor_sizes': [[3.9, 1.6, 1.56]], 'anchor_rotations': [0, 1.57], 'anchor_bottom_heights': [-1.78], 'align_center': False, 'feature_map_stride': 8, 'matched_threshold': 0.6, 'unmatched_threshold': 0.45}], 'TARGET_ASSIGNER_CONFIG': {'NAME': 'AxisAlignedTargetAssigner', 'POS_FRACTION': -1.0, 'SAMPLE_SIZE': 512, 'NORM_BY_NUM_EXAMPLES': False, 'MATCH_HEIGHT': False, 'BOX_CODER': 'ResidualCoder'}, 'LOSS_CONFIG': {'LOSS_WEIGHTS': {'cls_weight': 1.0, 'loc_weight': 2.0, 'dir_weight': 0.2, 'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}}}

        self.num_anchors_per_location = sum(self.num_anchors_per_location) #2

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d'] #torch.Size([8, 256, 200, 176])

        cls_preds = self.conv_cls(spatial_features_2d) #Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1)) torch.Size([8, 2, 200, 176])
        box_preds = self.conv_box(spatial_features_2d) #Conv2d(256, 14, kernel_size=(1, 1), stride=(1, 1)) torch.Size([8, 14, 200, 176])

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C] torch.Size([8, 200, 176, 2])
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C] torch.Size([8, 200, 176, 14])

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None: #Conv2d(256, 4, kernel_size=(1, 1), stride=(1, 1))
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d) #torch.Size([8, 4, 200, 176])
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous() #torch.Size([8, 200, 176, 4])
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )##'box_cls_labels':torch.Size([8, 70400])  'box_reg_targets':torch.Size([8, 70400, 7])  'reg_weights':torch.Size([8, 70400])
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            ) #batch_cls_preds： torch.Size([8, 70400, 1])  ，， batch_box_preds： torch.Size([8, 70400, 7])
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
