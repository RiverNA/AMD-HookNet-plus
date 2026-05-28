import torch
import torch.nn as nn
import copy
from parts import *


class AMD_HookNet_plus(nn.Module):
    def __init__(self, n_channels, filter_size=3, n_filters=64, config=True, img_size=224, num_classes=21843):
        super(AMD_HookNet_plus, self).__init__()
        self.n_channels = n_channels
        self.filter_size = filter_size
        self.n_filters = n_filters
        self.num_classes = num_classes
        self.config = config

        # Context branch
        self.swin_unet = SwinTransformerSys(img_size=config.DATA.IMG_SIZE,
                                            patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                            in_chans=config.MODEL.SWIN.IN_CHANS,
                                            num_classes=self.num_classes,
                                            embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                            depths=config.MODEL.SWIN.DEPTHS,
                                            num_heads=config.MODEL.SWIN.NUM_HEADS,
                                            window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                            qk_scale=config.MODEL.SWIN.QK_SCALE,
                                            drop_rate=config.MODEL.DROP_RATE,
                                            drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                            ape=config.MODEL.SWIN.APE,
                                            patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                            use_checkpoint=config.TRAIN.USE_CHECKPOINT)

        # Target branch
        self.t_first = Conv_block(n_channels, n_filters)
        self.t_down1 = Downsample(n_filters, n_filters * 2)
        self.t_down2 = Downsample(n_filters * 2, n_filters * 4)
        self.t_down3 = Downsample(n_filters * 4, n_filters * 8)
        self.t_down4 = Downsample(n_filters * 8, n_filters * 10)
        self.t_up1 = T_Upsample(n_filters * 16, n_filters * 8, HW=196)
        self.t_up2 = T_Upsample(n_filters * 11, n_filters * 4, HW=784)
        self.t_up3 = Upsample(n_filters * 4, n_filters * 2)
        self.t_up4 = Upsample(n_filters * 2, n_filters)
        self.t_out = Output(n_filters, self.num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            trunc_normal_(m.weight, std=(2.0 / fan_out) ** 0.5)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def load_from(self, config):
        pretrained_path = config.MODEL.RESUME
        if pretrained_path is not None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        del pretrained_dict[k]
                self.swin_unet.load_state_dict(pretrained_dict, strict=False)
                return
            pretrained_dict = pretrained_dict['model']
            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})

            for k, v in pretrained_dict.items():
                if "patch_embed." in k:
                    current_k = "patch_embed." + k[12:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        del full_dict[k]
            self.swin_unet.load_state_dict(full_dict, strict=False)
        else:
            print("none pretrain")

    def forward(self, x, y):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if y.size()[1] == 1:
            y = y.repeat(1, 3, 1, 1)
        x, c_outhooks = self.swin_unet(x)
        t_residuals = []
        y = self.t_first(y)
        t_residuals.append(y)
        y = self.t_down1(y)
        t_residuals.append(y)
        y = self.t_down2(y)
        t_residuals.append(y)
        y = self.t_down3(y)
        t_residuals.append(y)
        y = self.t_down4(y)
        z = []
        y, yy = self.t_up1(y, t_residuals[-1], c_outhooks[0])
        z.append(yy)
        y, yy = self.t_up2(y, t_residuals[-2], c_outhooks[1])
        z.append(yy)
        y = self.t_up3(y, t_residuals[-3])
        y = self.t_up4(y, t_residuals[-4])
        y = self.t_out(y)

        return x, y, z
