from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
from .voxel_rcnn import VoxelRCNN

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'VoxelRCNN': VoxelRCNN
}


def build_detector(model_cfg, num_class, dataset): #model_cfg: {'NAME': 'VoxelRCNN', 'VFE': {'NAME': 'MeanVFE'}, 'BACKBONE_3D': {'NAME': 'VoxelBackBone8x'}, 'MAP_TO_BEV': {'NAME': 'HeightCompression', 'NUM_BEV_FEATURES': 256}, 'BACKBONE_2D': {'NAME': 'BaseBEVBackbone', 'LAYER_NUMS': [4, 4], 'LAYER_STRIDES': [1, 2], 'NUM_FILTERS': [64, 128], 'UPSAMPLE_STRIDES': [1, 2], 'NUM_UPSAMPLE_FILTERS': [128, 128]}, 'DENSE_HEAD': {'NAME': 'AnchorHeadSingle', 'CLASS_AGNOSTIC': False, 'USE_DIRECTION_CLASSIFIER': True, 'DIR_OFFSET': 0.78539, 'DIR_LIMIT_OFFSET': 0.0, 'NUM_DIR_BINS': 2, 'ANCHOR_GENERATOR_CONFIG': [{'class_name': 'Car', 'anchor_sizes': [[3.9, 1.6, 1.56]], 'anchor_rotations': [0, 1.57], 'anchor_bottom_heights': [-1.78], 'align_center': False, 'feature_map_stride': 8, 'matched_threshold': 0.6, 'unmatched_threshold': 0.45}], 'TARGET_ASSIGNER_CONFIG': {'NAME': 'AxisAlignedTargetAssigner', 'POS_FRACTION': -1.0, 'SAMPLE_SIZE': 512, 'NORM_BY_NUM_EXAMPLES': False, 'MATCH_HEIGHT': False, 'BOX_CODER': 'ResidualCoder'}, 'LOSS_CONFIG': {'LOSS_WEIGHTS': {'cls_weight': 1.0, 'loc_weight': 2.0, 'dir_weight': 0.2, 'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}}}, 'ROI_HEAD': {'NAME': 'VoxelRCNNHead', 'CLASS_AGNOSTIC': True, 'SHARED_FC': [256, 256], 'CLS_FC': [256, 256], 'REG_FC': [256, 256], 'DP_RATIO': 0.3, 'NMS_CONFIG': {'TRAIN': {'NMS_TYPE': 'nms_gpu', 'MULTI_CLASSES_NMS': False, 'NMS_PRE_MAXSIZE': 9000, 'NMS_POST_MAXSIZE': 512, 'NMS_THRESH': 0.8}, 'TEST': {'NMS_TYPE': 'nms_gpu', 'MULTI_CLASSES_NMS': False, 'USE_FAST_NMS': True, 'SCORE_THRESH': 0.0, 'NMS_PRE_MAXSIZE': 2048, 'NMS_POST_MAXSIZE': 100, 'NMS_THRESH': 0.7}}, 'ROI_GRID_POOL': {'FEATURES_SOURCE': ['x_conv3', 'x_conv4'], 'PRE_MLP': True, 'GRID_SIZE': 6, 'POOL_LAYERS': {'x_conv3': {'MLPS': [[32, 32], [32, 32]], 'QUERY_RANGES': [[2, 2, 2], [4, 4, 4]], 'POOL_RADIUS': [0.4, 0.8], 'NSAMPLE': [16, 16], 'POOL_METHOD': 'max_pool'}, 'x_conv4': {'MLPS': [[32, 32], [32, 32]], 'QUERY_RANGES': [[2, 2, 2], [4, 4, 4]], 'POOL_RADIUS': [0.8, 1.6], 'NSAMPLE': [16, 16], 'POOL_METHOD': 'max_pool'}}}, 'TARGET_CONFIG': {'BOX_CODER': 'ResidualCoder', 'ROI_PER_IMAGE': 128, 'FG_RATIO': 0.5, 'SAMPLE_ROI_BY_EACH_CLASS': True, 'CLS_SCORE_TYPE': 'roi_iou', 'CLS_FG_THRESH': 0.75, 'CLS_BG_THRESH': 0.25, 'CLS_BG_THRESH_LO': 0.1, 'HARD_BG_RATIO': 0.8, 'REG_FG_THRESH': 0.55}, 'LOSS_CONFIG': {'CLS_LOSS': 'BinaryCrossEntropy', 'REG_LOSS': 'smooth-l1', 'CORNER_LOSS_REGULARIZATION': True, 'GRID_3D_IOU_LOSS': False, 'LOSS_WEIGHTS': {'rcnn_cls_weight': 1.0, 'rcnn_reg_weight': 1.0, 'rcnn_corner_weight': 1.0, 'rcnn_iou3d_weight': 1.0, 'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}}}, 'POST_PROCESSING': {'RECALL_THRESH_LIST': [0.3, 0.5, 0.7], 'SCORE_THRESH': 0.3, 'OUTPUT_RAW_SCORE': False, 'EVAL_METRIC': 'kitti', 'NMS_CONFIG': {'MULTI_CLASSES_NMS': False, 'NMS_TYPE': 'nms_gpu', 'NMS_THRESH': 0.1, 'NMS_PRE_MAXSIZE': 4096, 'NMS_POST_MAXSIZE': 500}}}
    model = __all__[model_cfg.NAME]( # VoxelRCNN
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
