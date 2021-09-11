import logging
import weakref
import time
import torch
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm
from detectron2.engine import DefaultTrainer, create_ddp_model, SimpleTrainer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import PascalVOCDetectionEvaluator, verify_results

import os
import sys
sys.path.append(os.getcwd())

from faster_rcnn.data.build import build_DA_detection_train_loader
from faster_rcnn.evaluation.test import PascalVOCDetectionEvaluator_

# register datasets
import faster_rcnn.data.register

# register compoments
from faster_rcnn.meta_archs.swda_rcnn import SWDARCNN
from faster_rcnn.backbones.fpn_da import build_resnet_fpn_da_backbone
from faster_rcnn.da_head.da_head import AlignmentHead
from faster_rcnn.da_head.da_roi_head import DAStandardROIHeads


class _DATrainer(SimpleTrainer):
    # one2one domain adpatation trainer
    def __init__(self, model, source_domain_data_loader, target_domain_data_loader, optimizer):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
        """
        super(SimpleTrainer).__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        model.train()

        self.model = model
        self.source_domain_data_loader = source_domain_data_loader
        self.target_domain_data_loader = target_domain_data_loader
        self._source_domain_data_loader_iter = iter(source_domain_data_loader)
        self._target_domain_data_loader_iter = iter(target_domain_data_loader)
        self.optimizer = optimizer

    def run_step(self):
        assert self.model.training, "[DASimpleTrainer] model was changed to eval mode!"
        loss_weight =  {'loss_cls':1, 'loss_box_reg': 1, 'loss_rpn_cls': 1, 'loss_rpn_loc': 1, \
            'local_alignment_loss': 0.5, 'global_alignment_loss': 0.5, \
        }

        start = time.perf_counter()
        data = next(self._source_domain_data_loader_iter)
        data_time = time.perf_counter() - start

        # source domain data loss
        loss_dict_source = self.model(data, input_domain='source')

        start = time.perf_counter()
        data = next(self._target_domain_data_loader_iter)
        data_time = time.perf_counter() - start + data_time

        # target domain data loss
        loss_dict_target = self.model(data, input_domain='target')

        loss_dict = {l: loss_weight[l] * (loss_dict_source[l] + loss_dict_target[l]) if loss_dict_target.get(l) else loss_weight[l] * loss_dict_source[l] for l in loss_weight}
        losses = sum(loss_dict.values())
        self.optimizer.zero_grad()
        losses.backward()

        self._write_metrics(loss_dict, data_time)

        self.optimizer.step()


class DATrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        source_domain_data_loader = self.build_train_loader(cfg, 'source')
        target_domain_data_loader = self.build_train_loader(cfg, 'target')


        model = create_ddp_model(model, broadcast_buffers=False)

        self._trainer = _DATrainer(
            model, source_domain_data_loader, target_domain_data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())
    
    @classmethod
    def build_train_loader(cls, cfg, dataset_domain):
        if dataset_domain == 'source': 
            return build_DA_detection_train_loader(cfg, dataset_domain=dataset_domain)
        elif dataset_domain == 'target':
            return build_DA_detection_train_loader(cfg, dataset_domain=dataset_domain)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return PascalVOCDetectionEvaluator_(dataset_name)

def add_swdarcnn_config(cfg):
    from detectron2.config import CfgNode as CN
    _C = cfg
    _C.DA_HEAD = CN()
    _C.DA_HEAD.DOMAIN_ADAPTATION_ON = True
    _C.DA_HEAD.GAMMA = 5.0
    _C.MODEL.ROI_HEADS.REGULARIZATION_FEATURES = ['local feature','global feature']
    _C.DATASETS.SOURCE_DOMAIN = CN()
    _C.DATASETS.SOURCE_DOMAIN.TRAIN = ()
    _C.DATASETS.TARGET_DOMAIN = CN()
    _C.DATASETS.TARGET_DOMAIN.TRAIN = ()



def setup(args):
    cfg = get_cfg()
    add_swdarcnn_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = DATrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = DATrainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = DATrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
