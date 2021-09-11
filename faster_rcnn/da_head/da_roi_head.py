# To do Context Vector based Regularization, 
# fuse Context Vector to feature outputed by convnet before localization branch nad classification branch
from typing import Dict, List, Optional
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import StandardROIHeads, ROI_HEADS_REGISTRY, FastRCNNOutputLayers, build_box_head
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Instances, Boxes
from detectron2.config import configurable
import torch
import torch.nn as nn

@ROI_HEADS_REGISTRY.register()
class DAROIHeads(StandardROIHeads):
    @configurable
    def __init__(self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        context_regularization_feat: List[str],
        context_regularization_on: bool,
        mask_in_features: Optional[List[str]] = None,
        mask_pooler: Optional[ROIPooler] = None,
        mask_head: Optional[nn.Module] = None,
        keypoint_in_features: Optional[List[str]] = None,
        keypoint_pooler: Optional[ROIPooler] = None,
        keypoint_head: Optional[nn.Module] = None,
        train_on_pred_boxes: bool = False,
        **kwargs,):

        super().__init__(box_in_features=box_in_features, box_pooler=box_pooler, box_head=box_head, box_predictor=box_predictor, \
            mask_in_features=mask_in_features, mask_pooler=mask_pooler, mask_head=mask_head, keypoint_in_features=keypoint_in_features, \
            keypoint_pooler=keypoint_pooler, keypoint_head=keypoint_head, train_on_pred_boxes=train_on_pred_boxes, \
            **kwargs,)
        self.context_regularization_feat = context_regularization_feat
        self.context_regularization_on = context_regularization_on

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape) 
        ret["context_regularization_feat"] = cfg.MODEL.ROI_HEADS.CONTEXT_REGULARIZATION_FEATURES
        ret["context_regularization_on"] = cfg.MODEL.ROI_HEADS.CONTEXT_REGULARIZATION_ON
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        # to define FastRCNNOutputLayers input shape, concat with context regularization feature(global feat, local feat)
        n = 1
        if cfg.MODEL.ROI_HEADS.CONTEXT_REGULARIZATION_ON:
            if cfg.DA_HEADS.GLOBAL_ALIGNMENT_ON:
                n += 1 
            if cfg.DA_HEADS.LOCAL_ALIGNMENT_ON:
                n += 1
        box_predictor = FastRCNNOutputLayers(cfg, ShapeSpec(channels=n * box_head.output_shape.channels))
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances]):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.
        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        # context vector regularization, local and global alignment features
        if self.context_regularization_on:
            reg_feat = [features[f].view(1,-1) for f in self.context_regularization_feat if not isinstance(features.get(f), type(None)) ]

        # filter feat by self.box_in_features
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)

        # fusion
        if self.context_regularization_on:
            box_features = torch.cat([f.repeat(box_features.size(0), 1) for f in reg_feat] + [box_features], 1)

        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances
