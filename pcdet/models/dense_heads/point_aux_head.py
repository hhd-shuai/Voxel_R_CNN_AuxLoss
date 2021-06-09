import torch

from ...utils import box_coder_utils, box_utils
from .point_head_template import PointHeadTemplate
from ...ops.pointnet2.pointnet2_original import pointnet2_utils


class PointAuxHead(PointHeadTemplate):
    """
    A simple point-based segmentation head, which are used for PointRCNN.
    Reference Paper: https://arxiv.org/abs/1812.04244
    PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud
    """
    def __init__(self, num_class, model_cfg,**kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)

        self.point_fc_layers = self.make_fc_layers(
            fc_cfg = self.model_cfg.POINT_FC,
            input_channels = 160,
            output_channels = 64
        )
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC, #[256, 256]
            input_channels=64,
            output_channels=1
        )
        self.reg_layers = self.make_fc_layers(
            fc_cfg = self.model_cfg.REG_FC,
            input_channels = 64,
            output_channels = 3
        )


    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            center_offset_target: (N1 + N2 + N3 + ..., 3)
        """
        point_coords = input_dict['points_mean'] #torch.Size([124898, 4])
        gt_boxes = input_dict['gt_boxes'] #torch.Size([8, 17, 8])
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1]) #torch.Size([8, 17, 8])
        targets_dict = self.assign_aux_targets(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
        ) #'point_cls_labels': 124898,,'center_offset_target':torch.Size([124898, 3])

        return targets_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        aux_loss, tb_dict_1 = self.get_aux_layer_loss()

        tb_dict.update(tb_dict_1)
        return aux_loss, tb_dict

    def tensor2points(self, tensor, offset=(0., -40., -3.), voxel_size=(.05, .05, .1)):
        indices = tensor.indices.float()
        offset = torch.Tensor(offset).to(indices.device)
        voxel_size = torch.Tensor(voxel_size).to(indices.device)
        indices[:, 1:] = indices[:, [3, 2, 1]] * voxel_size + offset + .5 * voxel_size
        return tensor.features, indices

    def nearest_neighbor_interpolate(self, unknown, known, known_feats):
        """
        :param pts: (n, 4) tensor of the bxyz positions of the unknown features    torch.Size([123397, 4])
        :param ctr: (m, 4) tensor of the bxyz positions of the known features      torch.Size([230645, 4])
        :param ctr_feats: (m, C) tensor of features to be propigated                torch.Size([230645, 32])
        :return:
            new_features: (n, C) tensor of the features of the unknown features
        """
        dist, idx = pointnet2_utils.three_nn(unknown,known)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=1, keepdim=True)
        weight = dist_recip / norm
        interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)

        return interpolated_feats

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:

                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
        """
        if self.training:
            voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords'] #voxel_features: torch.Size([124459, 4]) ,,, voxel_coords:torch.Size([124459, 4])
            points_mean = torch.zeros_like(voxel_features) #torch.Size([124459, 4])
            points_mean[:, 0] = voxel_coords[:, 0]
            points_mean[:, 1:] = voxel_features[:, :3]

            vx_feat, vx_nxyz = self.tensor2points(batch_dict['multi_scale_3d_features']['x_conv2'], self.model_cfg.TARGET_CONFIG.POINT_CLOUD_ORIGIN, self.model_cfg.TARGET_CONFIG.INTERPOLATE_VOXEL_SIZE1) # vx_feat: torch.Size([230487, 32])  ,, vx_nxyz: torch.Size([230487, 4])
            p0 = self.nearest_neighbor_interpolate(points_mean, vx_nxyz, vx_feat) #torch.Size([124459, 32])

            vx_feat, vx_nxyz = self.tensor2points(batch_dict['multi_scale_3d_features']['x_conv3'], self.model_cfg.TARGET_CONFIG.POINT_CLOUD_ORIGIN, self.model_cfg.TARGET_CONFIG.INTERPOLATE_VOXEL_SIZE2) #vx_feat： torch.Size([157225, 64]), vx_nxyz：torch.Size([157225, 4])
            p1 = self.nearest_neighbor_interpolate(points_mean, vx_nxyz, vx_feat) #torch.Size([124459, 64])

            vx_feat, vx_nxyz = self.tensor2points(batch_dict['multi_scale_3d_features']['x_conv4'], self.model_cfg.TARGET_CONFIG.POINT_CLOUD_ORIGIN, self.model_cfg.TARGET_CONFIG.INTERPOLATE_VOXEL_SIZE3) #vx_feat： torch.Size([71605, 64]), vx_nxyz：torch.Size([71605, 4])
            p2 = self.nearest_neighbor_interpolate(points_mean, vx_nxyz, vx_feat) #torch.Size([124459, 64])

            pointwise = self.point_fc_layers(torch.cat([p0, p1, p2], dim=-1)) #torch.Size([124459, 64])
            point_cls_preds = self.cls_layers(pointwise) #torch.Size([124459, 1])
            center_offset_preds = self.reg_layers(pointwise) #torch.Size([124459, 3])

            point_cls_preds_max, _ = point_cls_preds.max(dim=-1)
            batch_dict['points_mean'] = points_mean
            batch_dict['point_cls_scores'] = torch.sigmoid(point_cls_preds_max) #124459 tensor([0.5071, 0.4905, 0.4091,  ..., 0.2657, 0.4850, 0.5114], device='cuda:0',grad_fn=<SigmoidBackward>)
            batch_dict['center_offset_preds'] = center_offset_preds #torch.Size([124459, 3])

            ret_dict = {'point_cls_preds': point_cls_preds,
                        'point_cls_labels':None,
                        'center_offset_preds': center_offset_preds,
                        'center_offset_target': None}

            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            ret_dict['center_offset_target'] = targets_dict['center_offset_target']


            self.forward_ret_dict = ret_dict

            return batch_dict

        else:
            return batch_dict


