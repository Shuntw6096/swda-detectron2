import logging
import weakref
import time
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer, create_ddp_model, SimpleTrainer
from detectron2.checkpoint import DetectionCheckpointer
from .evaluation.test import PascalVOCDetectionEvaluator_
from .data.build import build_DA_detection_train_loader

class _DATrainer(SimpleTrainer):
    # one2one domain adpatation trainer
    def __init__(self, model, source_domain_data_loader, target_domain_data_loader, loss_weight, optimizer):
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
        self.loss_weight = loss_weight
        self.optimizer = optimizer

    def run_step(self):
        assert self.model.training, "[DASimpleTrainer] model was changed to eval mode!"

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

        loss_dict = {l: self.loss_weight[l] * (loss_dict_source[l] + loss_dict_target[l]) if loss_dict_target.get(l) else self.loss_weight[l] * loss_dict_source[l] for l in self.loss_weight}
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

        loss_weight = {'loss_cls': 1, 'loss_box_reg': 1, 'loss_rpn_cls': 1, 'loss_rpn_loc': 1, \
            'loss_local_alignment': 0.5, 'loss_global_alignment': 0.5, \
        }
        if not cfg.DA_HEADS.LOCAL_ALIGNMENT_ON:
            loss_weight.pop('loss_local_alignment')
        if not cfg.DA_HEADS.GLOBAL_ALIGNMENT_ON:
            loss_weight.pop('loss_global_alignment')
        self._trainer = _DATrainer(
            model, source_domain_data_loader, target_domain_data_loader, loss_weight, optimizer
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

# class FewShotTuner(DefaultTrainer):
#     # Target domain few-shot tuner
