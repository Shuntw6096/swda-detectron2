import torch.nn as nn
import torch
from typing import Dict, List, Optional, Tuple
from detectron2.config import configurable
from detectron2.structures import Instances
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling import Backbone, build_backbone, build_proposal_generator, build_roi_heads, META_ARCH_REGISTRY
from detectron2.utils.events import get_event_storage
from ..da_head.da_head import build_da_heads

@META_ARCH_REGISTRY.register()
class SWDARCNN(GeneralizedRCNN):
    # modified from https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/meta_arch/rcnn.py
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        da_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        super().__init__(backbone=backbone, proposal_generator=proposal_generator, roi_heads=roi_heads, \
            pixel_mean=pixel_mean, pixel_std=pixel_std, input_format=input_format, vis_period=vis_period, \
        )
        self.da_heads = da_heads

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "da_heads": build_da_heads(cfg),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], input_domain='source'):
        """
        training flow
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
            input_domain: str, source or target domain input 
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        
        features = self.backbone(images.tensor)
        resnet_feat = {
            'local_head_feature': features.pop('res2'), 
            'global_head_feature': features.pop('res4'), 
            'feature domain': input_domain,
            }
        if input_domain == 'source':
            reg_faeture, da_loss = self.da_heads(resnet_feat)

        elif input_domain == 'target':
            _, da_loss = self.da_heads(resnet_feat)
            losses = {}
            losses.update(da_loss)
            return losses

        features.update(reg_faeture)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(da_loss)
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        # context regularization vector
        resnet_feat = {
            'local_head_feature': features.pop('res2'), 
            'global_head_feature': features.pop('res4'), 
            'feature domain': None
            }
        reg_feat = self.da_heads(resnet_feat)
        features.update(reg_feat)
        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results