import numpy as np
import torch
import torch.nn as nn


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels): #{'NAME': 'BaseBEVBackbone', 'LAYER_NUMS': [4, 4], 'LAYER_STRIDES': [1, 2], 'NUM_FILTERS': [64, 128], 'UPSAMPLE_STRIDES': [1, 2], 'NUM_UPSAMPLE_FILTERS': [128, 128]}  ,,, 256
        super().__init__()
        self.model_cfg = model_cfg #{'NAME': 'BaseBEVBackbone', 'LAYER_NUMS': [4, 4], 'LAYER_STRIDES': [1, 2], 'NUM_FILTERS': [64, 128], 'UPSAMPLE_STRIDES': [1, 2], 'NUM_UPSAMPLE_FILTERS': [128, 128]}

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS #[4, 4]
            layer_strides = self.model_cfg.LAYER_STRIDES #[1, 2]
            num_filters = self.model_cfg.NUM_FILTERS  #[64, 128]
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS # [128, 128]
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES #[1, 2]
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums) #2
        c_in_list = [input_channels, *num_filters[:-1]] #[256, 64]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ] # [ZeroPad2d(padding=(1, 1, 1, 1), value=0.0), Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), bias=False), BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True), ReLU()]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ]) # [ZeroPad2d(padding=(1, 1, 1, 1), value=0.0), Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), bias=False), BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True), ReLU(), Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True), ReLU(), Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True), ReLU(), Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True), ReLU(), Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True), ReLU()]
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0: # upsample_strides: [1, 2]
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters) #256
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features'] #torch.Size([8, 256, 200, 176])
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x) #i = 0: torch.Size([8, 64, 200, 176]) ,,, i = 1: torch.Size([8, 128, 100, 88])

            stride = int(spatial_features.shape[2] / x.shape[2]) #1 ,, 2
            ret_dict['spatial_features_%dx' % stride] = x  #'spatial_features_1x': torch.Size([8, 64, 200, 176])  ,, 'spatial_features_2x': torch.Size([8, 128, 100, 88])
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1) #ups[0],ups[1]   torch.Size([8, 128, 200, 176])    -->  torch.Size([8, 256, 200, 176])
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x #torch.Size([8, 256, 200, 176])

        return data_dict
