from detectron2.modeling import FPN, BACKBONE_REGISTRY, build_resnet_backbone
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
import torch.nn.functional as F
from detectron2.layers import ShapeSpec

class DAFPN(FPN):
    # SWDA version FPN
    def forward(self, x):
        # modified from https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/fpn.py
        # output res2 and res4 feature to da-heads 
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        bottom_up_features = self.bottom_up(x)
        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        results.append(self.output_convs[0](prev_features))

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(
            zip(self.lateral_convs, self.output_convs)
        ):
            # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            # Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = self.in_features[-idx - 1]
                features = bottom_up_features[features]
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if self._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        # add res2 and res4 features to output, it will feed to DA head
        output = {f: res for f, res in zip(self._out_features, results)}
        output.update({'res2': bottom_up_features['res2'], 'res4': bottom_up_features['res4']})
        return output

@BACKBONE_REGISTRY.register()
def build_resnet_fpn_da_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = DAFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    # already frozen
    # backbone.apply(set_batchnorm_fix)
    return backbone

def set_batchnorm_fix(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        for p in m.parameters(): 
            p.requires_grad=False