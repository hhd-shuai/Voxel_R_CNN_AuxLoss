import torch
import torch.nn as nn
from .roi_head_template import RoIHeadTemplate
from ...utils import common_utils, spconv_utils
from ...ops.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules


class VoxelRCNNHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, point_cloud_range, voxel_size, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg #{'NAME': 'VoxelRCNNHead', 'CLASS_AGNOSTIC': True, 'SHARED_FC': [256, 256], 'CLS_FC': [256, 256], 'REG_FC': [256, 256], 'DP_RATIO': 0.3, 'NMS_CONFIG': {'TRAIN': {'NMS_TYPE': 'nms_gpu', 'MULTI_CLASSES_NMS': False, 'NMS_PRE_MAXSIZE': 9000, 'NMS_POST_MAXSIZE': 512, 'NMS_THRESH': 0.8}, 'TEST': {'NMS_TYPE': 'nms_gpu', 'MULTI_CLASSES_NMS': False, 'USE_FAST_NMS': True, 'SCORE_THRESH': 0.0, 'NMS_PRE_MAXSIZE': 2048, 'NMS_POST_MAXSIZE': 100, 'NMS_THRESH': 0.7}}, 'ROI_GRID_POOL': {'FEATURES_SOURCE': ['x_conv3', 'x_conv4'], 'PRE_MLP': True, 'GRID_SIZE': 6, 'POOL_LAYERS': {'x_conv3': {'MLPS': [[32, 32], [32, 32]], 'QUERY_RANGES': [[2, 2, 2], [4, 4, 4]], 'POOL_RADIUS': [0.4, 0.8], 'NSAMPLE': [16, 16], 'POOL_METHOD': 'max_pool'}, 'x_conv4': {'MLPS': [[32, 32], [32, 32]], 'QUERY_RANGES': [[2, 2, 2], [4, 4, 4]], 'POOL_RADIUS': [0.8, 1.6], 'NSAMPLE': [16, 16], 'POOL_METHOD': 'max_pool'}}}, 'TARGET_CONFIG': {'BOX_CODER': 'ResidualCoder', 'ROI_PER_IMAGE': 128, 'FG_RATIO': 0.5, 'SAMPLE_ROI_BY_EACH_CLASS': True, 'CLS_SCORE_TYPE': 'roi_iou', 'CLS_FG_THRESH': 0.75, 'CLS_BG_THRESH': 0.25, 'CLS_BG_THRESH_LO': 0.1, 'HARD_BG_RATIO': 0.8, 'REG_FG_THRESH': 0.55}, 'LOSS_CONFIG': {'CLS_LOSS': 'BinaryCrossEntropy', 'REG_LOSS': 'smooth-l1', 'CORNER_LOSS_REGULARIZATION': True, 'GRID_3D_IOU_LOSS': False, 'LOSS_WEIGHTS': {'rcnn_cls_weight': 1.0, 'rcnn_reg_weight': 1.0, 'rcnn_corner_weight': 1.0, 'rcnn_iou3d_weight': 1.0, 'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}}}
        self.pool_cfg = model_cfg.ROI_GRID_POOL #{'FEATURES_SOURCE': ['x_conv3', 'x_conv4'], 'PRE_MLP': True, 'GRID_SIZE': 6, 'POOL_LAYERS': {'x_conv3': {'MLPS': [[32, 32], [32, 32]], 'QUERY_RANGES': [[2, 2, 2], [4, 4, 4]], 'POOL_RADIUS': [0.4, 0.8], 'NSAMPLE': [16, 16], 'POOL_METHOD': 'max_pool'}, 'x_conv4': {'MLPS': [[32, 32], [32, 32]], 'QUERY_RANGES': [[2, 2, 2], [4, 4, 4]], 'POOL_RADIUS': [0.8, 1.6], 'NSAMPLE': [16, 16], 'POOL_METHOD': 'max_pool'}}}
        LAYER_cfg = self.pool_cfg.POOL_LAYERS #{'x_conv3': {'MLPS': [[32, 32], [32, 32]], 'QUERY_RANGES': [[2, 2, 2], [4, 4, 4]], 'POOL_RADIUS': [0.4, 0.8], 'NSAMPLE': [16, 16], 'POOL_METHOD': 'max_pool'}, 'x_conv4': {'MLPS': [[32, 32], [32, 32]], 'QUERY_RANGES': [[2, 2, 2], [4, 4, 4]], 'POOL_RADIUS': [0.8, 1.6], 'NSAMPLE': [16, 16], 'POOL_METHOD': 'max_pool'}}
        self.point_cloud_range = point_cloud_range #[  0.  -40.   -3.   70.4  40.    1. ]
        self.voxel_size = voxel_size #[0.05, 0.05, 0.1]

        c_out = 0  #64
        self.roi_grid_pool_layers = nn.ModuleList()
        for src_name in self.pool_cfg.FEATURES_SOURCE:
            mlps = LAYER_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [input_channels[src_name]] + mlps[k]  #--> [[64, 32, 32], [64, 32, 32]]
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                query_ranges=LAYER_cfg[src_name].QUERY_RANGES,
                nsamples=LAYER_cfg[src_name].NSAMPLE,
                radii=LAYER_cfg[src_name].POOL_RADIUS,
                mlps=mlps,
                pool_method=LAYER_cfg[src_name].POOL_METHOD,
            )
            
            self.roi_grid_pool_layers.append(pool_layer)

            c_out += sum([x[-1] for x in mlps])


        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE # 6
        # c_out = sum([x[-1] for x in mlps])
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out #27648

        shared_fc_list = [] #[Linear(in_features=27648, out_features=256, bias=False), BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(inplace=True), Dropout(p=0.3, inplace=False), Linear(in_features=256, out_features=256, bias=False), BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(inplace=True)]
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.SHARED_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU(inplace=True)
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        cls_fc_list = [] #[Linear(in_features=256, out_features=256, bias=False), BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(), Dropout(p=0.3, inplace=False), Linear(in_features=256, out_features=256, bias=False), BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU()]
        for k in range(0, self.model_cfg.CLS_FC.__len__()):
            cls_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.CLS_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.CLS_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.CLS_FC[k]

            if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                cls_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.cls_fc_layers = nn.Sequential(*cls_fc_list)
        self.cls_pred_layer = nn.Linear(pre_channel, self.num_class, bias=True)

        reg_fc_list = [] #[Linear(in_features=256, out_features=256, bias=False), BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(), Dropout(p=0.3, inplace=False), Linear(in_features=256, out_features=256, bias=False), BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU()]
        for k in range(0, self.model_cfg.REG_FC.__len__()):
            reg_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.REG_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.REG_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.REG_FC[k]

            if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.reg_fc_layers = nn.Sequential(*reg_fc_list)
        self.reg_pred_layer = nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias=True)

        self.init_weights()

    def init_weights(self):
        init_func = nn.init.xavier_normal_
        for module_list in [self.shared_fc_layer, self.cls_fc_layers, self.reg_fc_layers]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                    
        nn.init.normal_(self.cls_pred_layer.weight, 0, 0.01)
        nn.init.constant_(self.cls_pred_layer.bias, 0)
        nn.init.normal_(self.reg_pred_layer.weight, mean=0, std=0.001)
        nn.init.constant_(self.reg_pred_layer.bias, 0)

    # def _init_weights(self):
    #     init_func = nn.init.xavier_normal_
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
    #             init_func(m.weight)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #     nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)
    
    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        rois = batch_dict['rois'] #torch.Size([8, 128, 7])
        batch_size = batch_dict['batch_size']
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False) #False
        
        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size=self.pool_cfg.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        # roi_grid_xyz: (B, Nx6x6x6, 3)  torch.Size([1024, 216, 3])
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)   #torch.Size([8, 27648, 3])

        # compute the voxel coordinates of grid points
        roi_grid_coords_x = (roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]) // self.voxel_size[0] #torch.Size([8, 27648, 1])
        roi_grid_coords_y = (roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]) // self.voxel_size[1] #torch.Size([8, 27648, 1])
        roi_grid_coords_z = (roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]) // self.voxel_size[2] #torch.Size([8, 27648, 1])
        # roi_grid_coords: (B, Nx6x6x6, 3)
        roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1) #torch.Size([8, 27648, 3])

        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1) #torch.Size([8, 27648, 1])
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
        # roi_grid_coords: (B, Nx6x6x6, 4)
        # roi_grid_coords = torch.cat([batch_idx, roi_grid_coords], dim=-1)
        # roi_grid_coords = roi_grid_coords.int()
        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_coords.shape[1]) #8  tensor([27648, 27648, 27648, 27648, 27648, 27648, 27648, 27648],device='cuda:0', dtype=torch.int32)

        pooled_features_list = []
        for k, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE): #self.pool_cfg.FEATURES_SOURCE: ['x_conv3', 'x_conv4']
            pool_layer = self.roi_grid_pool_layers[k]
            cur_stride = batch_dict['multi_scale_3d_strides'][src_name] #4
            cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

            if with_vf_transform:
                cur_sp_tensors = batch_dict['multi_scale_3d_features_post'][src_name]
            else:
                cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name] #[11, 400, 352]

            # compute voxel center xyz and batch_cnt
            cur_coords = cur_sp_tensors.indices #torch.Size([157704, 4])
            cur_voxel_xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=cur_stride,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            ) #torch.Size([157704, 3])
            cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()  #8  tensor([0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0', dtype=torch.int32)
            for bs_idx in range(batch_size):
                cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
            # get voxel2point tensor
            v2p_ind_tensor = spconv_utils.generate_voxel2pinds(cur_sp_tensors) #torch.Size([8, 11, 400, 352])
            # compute the grid coordinates in this scale, in [batch_idx, x y z] order
            cur_roi_grid_coords = roi_grid_coords // cur_stride #torch.Size([8, 27648, 3])
            cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1) #torch.Size([8, 27648, 4])
            cur_roi_grid_coords = cur_roi_grid_coords.int() #torch.Size([8, 27648, 4])
            # voxel neighbor aggregation !!!!!!!!
            pooled_features = pool_layer(
                xyz=cur_voxel_xyz.contiguous(),
                xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
                new_xyz_batch_cnt=roi_grid_batch_cnt,
                new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                features=cur_sp_tensors.features.contiguous(),
                voxel2point_indices=v2p_ind_tensor
            ) #torch.Size([221184, 64])

            pooled_features = pooled_features.view(
                -1, self.pool_cfg.GRID_SIZE ** 3,
                pooled_features.shape[-1]
            )  # (BxN, 6x6x6, C)     torch.Size([1024, 216, 64])
            pooled_features_list.append(pooled_features)
        
        ms_pooled_features = torch.cat(pooled_features_list, dim=-1) #torch.Size([1024, 216, 128])
        
        return ms_pooled_features


    def get_global_grid_points_of_roi(self, rois, grid_size): #rois：torch.Size([1024, 7]) grid_size： 6
        rois = rois.view(-1, rois.shape[-1]) #torch.Size([1024, 7])
        batch_size_rcnn = rois.shape[0] #1024

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3) torch.Size([1024, 216, 3])
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1) #torch.Size([1024, 216, 3])
        global_center = rois[:, 0:3].clone() #torch.Size([1024, 3])
        global_roi_grid_points += global_center.unsqueeze(dim=1)#torch.Size([1024, 216, 3])
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size): #rois: torch.Size([1024, 7]) ，，batch_size_rcnn：1024，， grid_size：6
        faked_features = rois.new_ones((grid_size, grid_size, grid_size)) #torch.Size([6, 6, 6])
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx] torch.Size([216, 3])
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3) torch.Size([1024, 216, 3])

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6] #torch.Size([1024, 3])
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3) torch.Size([1024, 216, 3])
        return roi_grid_points

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        ) #roi
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois'] #torch.Size([8, 128, 7])
            batch_dict['roi_labels'] = targets_dict['roi_labels'] #torch.Size([8, 128])

        # RoI aware pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)    torch.Size([1024, 216, 128])

        # Box Refinement
        pooled_features = pooled_features.view(pooled_features.size(0), -1) #torch.Size([1024, 27648])
        shared_features = self.shared_fc_layer(pooled_features) #torch.Size([1024, 256])
        rcnn_cls = self.cls_pred_layer(self.cls_fc_layers(shared_features)) #torch.Size([1024, 1])
        rcnn_reg = self.reg_pred_layer(self.reg_fc_layers(shared_features)) #torch.Size([1024, 7])

        # grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        # batch_size_rcnn = pooled_features.shape[0]
        # pooled_features = pooled_features.permute(0, 2, 1).\
        #     contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        # shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        # rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        # rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict
