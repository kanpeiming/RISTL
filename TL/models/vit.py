import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.clock_driven import functional
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from utils.loss_function import linear_CKA

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.c_hidden = hidden_features
        self.c_output = out_features
    def forward(self, x):
        T,B,C,N = x.shape
        x = self.fc1_conv(x.flatten(0,1))
        x = self.fc1_bn(x).reshape(T,B,self.c_hidden,N).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_conv(x.flatten(0,1))
        x = self.fc2_bn(x).reshape(T,B,C,N).contiguous()
        x = self.fc2_lif(x)
        return x

class SSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.25

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.attn_drop = nn.Dropout(0.2)
        self.res_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T,B,C,N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T,B,C,N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T,B,C,N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        # 恢复原始注意力机制处理逻辑，参考cifar10dvs实现
        attn = (q @ k.transpose(-2, -1))
        x = (attn @ v) * self.scale
        
        # The CKA feature is calculated on the raw attention map
        attn_feature = self.attn_lif(attn * self.scale)

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.proj_lif(self.proj_bn(self.proj_conv(x.flatten(0,1))).reshape(T,B,C,N))

        return x, attn_feature

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        attn_out, attn_feature = self.attn(x)
        x = x + attn_out
        x = x + (self.mlp(x))
        return x, attn_feature

class SPS(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256, T=4):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.T = T

        # --- Add an initial projection layer to unify channel dimensions ---
        self.channel_proj = nn.Conv2d(in_channels, embed_dims, kernel_size=1, stride=1, padding=0, bias=False)
        self.channel_proj_bn = nn.BatchNorm2d(embed_dims)
        self.channel_proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        
        # --- Subsequent layers now operate on a unified `embed_dims` ---
        self.proj_conv = nn.Conv2d(embed_dims, embed_dims//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims//8)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv1 = nn.Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims//4)
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv2 = nn.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims//2)
        self.proj_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = x.flatten(0, 1)

        # --- Apply the initial channel projection ---
        x = self.channel_proj(x)
        x = self.channel_proj_bn(x).reshape(T, B, -1, H, W)
        x = self.channel_proj_lif(x)
        x = x.flatten(0,1)

        # Block 1
        x = self.proj_conv(x)
        x = self.proj_bn(x).reshape(T, B, -1, H, W)
        x = self.proj_lif(x)
        x = self.maxpool(x.flatten(0, 1))

        # Block 2
        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H // 2, W // 2)
        x = self.proj_lif1(x)
        x = self.maxpool1(x.flatten(0, 1))

        # Block 3
        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H // 4, W // 4)
        x = self.proj_lif2(x)
        x = self.maxpool2(x.flatten(0, 1))

        # Block 4
        x = self.proj_conv3(x)
        x = self.proj_bn3(x).reshape(T, B, -1, H // 8, W // 8)
        x = self.proj_lif3(x)
        x = self.maxpool3(x.flatten(0, 1))

        # RPE Block & Final Reshape
        x_rpe = self.rpe_conv(x)
        x_rpe = self.rpe_bn(x_rpe).reshape(T, B, -1, H // 16, W // 16)
        x_rpe = self.rpe_lif(x_rpe)

        x = x.reshape(T, B, -1, H // 16, W // 16) + x_rpe
        x = x.reshape(T, B, -1, (H // 16) * (W // 16))

        return x

class ViTSNN(nn.Module):
    def __init__(self, cls_num=10,
                 img_size_h=128, img_size_w=128, patch_size=16,
                 embed_dims=256, num_heads=16, mlp_ratios=4, qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=2, sr_ratios=1, T=4):
        super().__init__()

        self.rgb_input_encoder = SPS(img_size_h=img_size_h, img_size_w=img_size_w, patch_size=patch_size, in_channels=3, embed_dims=embed_dims, T=T)
        self.dvs_input_encoder = SPS(img_size_h=img_size_h, img_size_w=img_size_w, patch_size=patch_size, in_channels=3, embed_dims=embed_dims, T=T)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]
        self.features = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])

        self.bottleneck = nn.Linear(embed_dims, 256)
        self.bottleneck_lif_node = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.classifier = nn.Linear(256, cls_num)
        self.loss_function = linear_CKA

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, source, target=None):
        if self.training:
            if target is None:
                raise ValueError("In training mode, both source (RGB) and target (DVS) are required.")

            # --- Input Preprocessing ---
            source = source.permute(1, 0, 2, 3, 4).contiguous()
            target = target.permute(1, 0, 2, 3, 4).contiguous()

            # --- SOURCE BRANCH (RGB) ---
            sps_source_out = self.rgb_input_encoder(source)

            source_features = sps_source_out
            source_attn_feature_last = None
            for i, blk in enumerate(self.features):
                source_features, attn_feature = blk(source_features)
                if i == len(self.features) - 1:
                    source_attn_feature_last = attn_feature
            block_last_source_out = source_features

            source_clf = source_features.mean(3)
            source_clf = self.bottleneck(source_clf)
            source_clf = self.bottleneck_lif_node(source_clf)
            source_clf = self.classifier(source_clf.mean(0))

            # --- Reset network state ---
            functional.reset_net(self)

            # --- TARGET BRANCH (DVS) ---
            sps_target_out = self.dvs_input_encoder(target)

            target_features = sps_target_out
            target_attn_feature_last = None
            for i, blk in enumerate(self.features):
                target_features, attn_feature = blk(target_features)
                if i == len(self.features) - 1:
                    target_attn_feature_last = attn_feature
            block_last_target_out = target_features

            target_clf = target_features.mean(3)
            target_clf = self.bottleneck(target_clf)
            target_clf = self.bottleneck_lif_node(target_clf)
            target_clf = self.classifier(target_clf.mean(0))

            # --- CKA Loss Calculation ---
            sps_source_reshaped = sps_source_out.reshape(sps_source_out.size(0) * sps_source_out.size(1), -1)
            sps_target_reshaped = sps_target_out.reshape(sps_target_out.size(0) * sps_target_out.size(1), -1)
            sps_cka_loss = self.loss_function(sps_source_reshaped, sps_target_reshaped)

            block_source_reshaped = block_last_source_out.reshape(block_last_source_out.size(0) * block_last_source_out.size(1), -1)
            block_target_reshaped = block_last_target_out.reshape(block_last_target_out.size(0) * block_last_target_out.size(1), -1)
            block_cka_loss = self.loss_function(block_source_reshaped, block_target_reshaped)

            attn_source_reshaped = source_attn_feature_last.reshape(source_attn_feature_last.size(0) * source_attn_feature_last.size(1) * source_attn_feature_last.size(2), -1)
            attn_target_reshaped = target_attn_feature_last.reshape(target_attn_feature_last.size(0) * target_attn_feature_last.size(1) * target_attn_feature_last.size(2), -1)
            attn_cka_loss = self.loss_function(attn_source_reshaped, attn_target_reshaped)

            return source_clf, target_clf, sps_cka_loss, block_cka_loss, attn_cka_loss
        else:
            # --- EVALUATION MODE ---
            if source.dim() == 4:
                source = source.unsqueeze(1).repeat(1, self.dvs_input_encoder.T, 1, 1, 1)
            
            eval_input = source.permute(1, 0, 2, 3, 4).contiguous()
            x = self.dvs_input_encoder(eval_input)

            for blk in self.features:
                x, _ = blk(x)
            x = x.mean(3)
            x = self.bottleneck(x)
            x = self.bottleneck_lif_node(x)
            target_clf = self.classifier(x.mean(0))
            return target_clf