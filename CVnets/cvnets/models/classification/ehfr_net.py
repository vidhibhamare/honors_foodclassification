
from torch import nn
import torch
import argparse
from typing import Dict, Tuple
from torchvision import models
from transformers import SwinModel

from . import register_cls_models
from .base_cls import BaseEncoder
from .config.ehfr_net import get_configuration
from ...layers import ConvLayer, LinearLayer, GlobalPool, Identity, cross_attention
from ...modules import HBlock as Block


@register_cls_models("ehfr_net")
class EHFR_Net(BaseEncoder):
    """
    This class defines the EHFR_Net architecture
    """
    
    def __init__(self, opts, *args, **kwargs) -> None:
        num_classes = getattr(opts, "model.classification.n_classes", 101)
        pool_type = getattr(opts, "model.layer.global_pool", "mean")

        # Initialize backbone (EfficientNet)
        super().__init__(*args, **kwargs)
        backbone = models.efficientnet_b0(pretrained=True)
        self.backbone = backbone.features
        cnn_out_channels = 1280  # EfficientNet-B0's final feature dimension

        # Swin Transformer initialization
        self.swin = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        
        # Projection layer to adapt CNN features to Swin input
        # self.projection = nn.Sequential(
        #     ConvLayer(
        #         opts=opts,
        #         in_channels=cnn_out_channels,
        #         out_channels=3,  # Swin expects 3-channel input
        #         kernel_size=1,
        #         use_norm=True,
        #         use_act=True
        #     ),
        #     nn.AdaptiveAvgPool2d((224, 224))  # Resize to Swin's expected input size
        # )
        # Add after Swin initialization
        self.fusion = cross_attention.CrossAttentionFusion(
            cnn_dim=1280,  # EfficientNet-B0 output channels
            vit_dim=768    # Swin-Tiny embed_dim
        )
        
        self.projection = nn.Sequential(
            nn.Conv2d(1280, 3, kernel_size=1),  # Channel reduction
            nn.AdaptiveAvgPool2d((224, 224)),   # Spatial resizing
            nn.Hardswish()                      # Activation
        )
        # Classification head
        # self.classifier = nn.Sequential(
        #     # GlobalPool(pool_type=pool_type, keep_dim=False),
        #     nn.LayerNorm(768), 
        #     LinearLayer(in_features=768, out_features=num_classes, bias=True),  # Swin base has 768-dim output
        # )
        # Modify classifier input dim
        # Change this in __init__
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            LinearLayer(768, num_classes)  # Input dim matches Swin
        )


        # Model configuration dictionary
        self.model_conf_dict = {
            "backbone": {"in": 3, "out": cnn_out_channels},
            "projection": {"in": cnn_out_channels, "out": 3},
            "classifier": {"in": 768, "out": num_classes}
        }

        self.reset_parameters(opts=opts)

    def forward(self, x):
        # Feature extraction
        cnn_feats = self.backbone(x)  # [B, 1280, H, W]
        projected = self.projection(cnn_feats)  # [B, 3, 224, 224]
        swin_out = self.swin(projected).last_hidden_state  # [B, 49, 768]
        
        # Cross-attention fusion
        fused = self.fusion(cnn_feats, swin_out)  # [B, 49, 768]
        
        # Classification
        cls_token = fused[:, 0]  # Take CLS token [B, 768]
        return self.classifier(cls_token)
    # def forward(self, x):
    #     # 1. EfficientNet feature extraction
    #     cnn_features = self.backbone(x)  # [B, 1280, H, W]
        
    #     # 2. Project to Swin input space
    #     projected_features = self.projection(cnn_features)  # [B, 3, 224, 224]
        
    #     # 3. Swin Transformer processing
    #     swin_output = self.swin(projected_features)  # BaseModelOutput
        
    #     # 4. Dimension handling with debug checks
    #     if not hasattr(swin_output, 'last_hidden_state'):
    #         raise ValueError("Swin output missing last_hidden_state. Got keys: " + str(swin_output.keys()))
        
    #     last_hidden = swin_output.last_hidden_state  # [B, seq_len, embed_dim]
        
    #     # 5. Proper CLS token extraction
    #     if last_hidden.dim() != 3:
    #         raise ValueError(f"Hidden state must be 3D. Got shape {last_hidden.shape}")
        
    #     cls_token = last_hidden[:, 0, :]  # Take first token [B, embed_dim]
        
    #     # 6. Ensure proper dimensions before classifier
    #     if cls_token.dim() == 1:
    #         cls_token = cls_token.unsqueeze(0)  # Add batch dim if missing
    #     elif cls_token.dim() != 2:
    #         raise ValueError(f"CLS token must be 2D. Got shape {cls_token.shape}")

    #     # 7. Classification
    #     return self.classifier(cls_token)
    #commented before swin
    # def __init__(self, opts, *args, **kwargs) -> None:
    #     num_classes = getattr(opts, "model.classification.n_classes", 101)
    #     pool_type = getattr(opts, "model.layer.global_pool", "mean")

    #     # First call parent's __init__()
    #     super().__init__(*args, **kwargs)

    #     # Then initialize backbone
    #     backbone = models.efficientnet_b0(pretrained=True)
    #     self.backbone = backbone.features
    #     out_channels = 1280

    #     # Rest of your initialization...
    #     self.model_conf_dict = {
    #         "backbone": {"in": 3, "out": out_channels},
    #         "exp_before_cls": {"in": out_channels, "out": out_channels}
    #     }
    #     self.conv_1x1_exp = Identity()
    #     self.classifier = nn.Sequential(
    #         GlobalPool(pool_type=pool_type, keep_dim=False),
    #         LinearLayer(in_features=out_channels, out_features=num_classes, bias=True),
    #     )
    #     self.reset_parameters(opts=opts)

    # def forward(self, x):
    #     # EfficientNet feature extraction
    #     x = self.backbone(x)
        
    #     # Original EHFR-Net processing
    #     x = self.conv_1x1_exp(x)
    #     x = self.classifier(x)
    #     return x
    
    
    
    
    # def __init__(self, opts, *args, **kwargs) -> None:
    #     num_classes = getattr(opts, "model.classification.n_classes", 101)
    #     pool_type = getattr(opts, "model.layer.global_pool", "mean")
        
    #     #new code starts
        
    #     # Replace manual CNN blocks with EfficientNet-B0
    #     self.backbone = models.efficientnet_b0(pretrained=True).features
        
    #     # Get output channels from EfficientNet's last layer
    #     out_channels = 1280  # EfficientNet-B0's final feature dimension
        
    #     self.conv_1x1_exp = Identity()
    #     self.model_conf_dict = {
    #         "exp_before_cls": {"in": out_channels, "out": out_channels}
    #     }
        
    #     self.classifier = nn.Sequential(
    #         GlobalPool(pool_type=pool_type, keep_dim=False),
    #         LinearLayer(in_features=out_channels, out_features=num_classes, bias=True),
    #     )
    #     #new code ends
        
    #     ehfr_net_config = get_configuration(opts=opts)
    #     image_channels = ehfr_net_config["layer0"]["img_channels"]
    #     out_channels = ehfr_net_config["layer0"]["out_channels"]

    #     super().__init__(*args, **kwargs)

    #     # store model configuration in a dictionary
    #     self.model_conf_dict = dict()
    #     self.conv_1 = ConvLayer(
    #         opts=opts,
    #         in_channels=image_channels,
    #         out_channels=out_channels,
    #         kernel_size=3,
    #         stride=2,
    #         use_norm=True,
    #         use_act=True,
    #     )

    #     self.model_conf_dict["conv1"] = {"in": image_channels, "out": out_channels}

    #     in_channels = out_channels
    #     self.layer_1, out_channels = self._make_hblock(
    #         opts=opts, input_channel=in_channels, cfg=ehfr_net_config["layer1"]
    #     )
    #     self.model_conf_dict["layer1"] = {"in": in_channels, "out": out_channels}

    #     in_channels = out_channels
    #     self.layer_2, out_channels = self._make_hblock(
    #         opts=opts, input_channel=in_channels, cfg=ehfr_net_config["layer2"]
    #     )
    #     self.model_conf_dict["layer2"] = {"in": in_channels, "out": out_channels}

    #     in_channels = out_channels
    #     self.layer_3, out_channels = self._make_hblock(
    #         opts=opts, input_channel=in_channels, cfg=ehfr_net_config["layer3"]
    #     )
    #     self.model_conf_dict["layer3"] = {"in": in_channels, "out": out_channels}

    #     in_channels = out_channels
    #     self.layer_4, out_channels = self._make_hblock(
    #         opts=opts, input_channel=in_channels, cfg=ehfr_net_config["layer4"],
    #     )
    #     self.model_conf_dict["layer4"] = {"in": in_channels, "out": out_channels}

    #     in_channels = out_channels
    #     self.layer_5, out_channels = self._make_hblock(
    #         opts=opts, input_channel=in_channels, cfg=ehfr_net_config["layer5"],
    #     )
    #     self.model_conf_dict["layer5"] = {"in": in_channels, "out": out_channels}

    #     self.conv_1x1_exp = Identity()
    #     self.model_conf_dict["exp_before_cls"] = {
    #         "in": out_channels,
    #         "out": out_channels,
    #     }

    #     self.classifier = nn.Sequential(
    #         GlobalPool(pool_type=pool_type, keep_dim=False),
    #         LinearLayer(in_features=out_channels, out_features=num_classes, bias=True),
    #     )

    #     # check model
    #     self.check_model()

    #     # weight initialization
    #     self.reset_parameters(opts=opts)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--model.classification.ehfr_net.attn-dropout",
            type=float,
            default=0.0,
            help="Dropout in attention layer. Defaults to 0.0",
        )
        group.add_argument(
            "--model.classification.ehfr_net.ffn-dropout",
            type=float,
            default=0.0,
            help="Dropout between FFN ehfr_net. Defaults to 0.0",
        )
        group.add_argument(
            "--model.classification.ehfr_net.dropout",
            type=float,
            default=0.0,
            help="Dropout in attention layer. Defaults to 0.0",
        )
        group.add_argument(
            "--model.classification.ehfr_net.width-multiplier",
            type=float,
            default=1.0,
            help="Width multiplier. Defaults to 1.0",
        )
        group.add_argument(
            "--model.classification.ehfr_net.attn-norm-layer",
            type=str,
            default="layer_norm_2d",
            help="Norm layer in attention block. Defaults to LayerNorm",
        )
        return parser

    def _make_hblock(
        self, opts, input_channel, cfg: Dict
    ) -> Tuple[nn.Sequential, int]:
        prev_dilation = self.dilation
        block = []

        ffn_multiplier = cfg.get("ffn_multiplier")

        dropout = getattr(opts, "model.classification.ehfr_net.dropout", 0.0)

        block.append(
            Block(
                opts=opts,
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=cfg.get("stride", 1),
                ffn_multiplier=ffn_multiplier,
                n_local_blocks=cfg.get("n_local_blocks", 1),
                n_attn_blocks=cfg.get("n_attn_blocks", 1),
                patch_h=cfg.get("patch_h", 2),
                patch_w=cfg.get("patch_w", 2),
                dropout=dropout,
                ffn_dropout=getattr(
                    opts, "model.classification.ehfr_net.ffn_dropout", 0.0
                ),
                attn_dropout=getattr(
                    opts, "model.classification.ehfr_net.attn_dropout", 0.0
                ),
                attn_norm_layer=getattr(
                    opts, "model.classification.ehfr_net.attn_norm_layer", "layer_norm_2d"
                ),
                expand_ratio=cfg.get("expand_ratio", 4),
                dilation=prev_dilation,
            )
        )

        input_channel = cfg.get("out_channels")

        return nn.Sequential(*block), input_channel
