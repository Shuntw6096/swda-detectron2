from detectron2.utils import comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch, DefaultPredictor
from detectron2.evaluation import verify_results

import os
import sys
from datetime import datetime
sys.path.append(os.getcwd())

from faster_rcnn.trainer import DATrainer, FewShotTuner

# register datasets
import faster_rcnn.data.register

# register compoments
from faster_rcnn.meta_archs.swda_rcnn import SWDARCNN
from faster_rcnn.backbones.fpn_da import build_resnet_fpn_da_backbone
from faster_rcnn.da_head.da_head import AlignmentHead
from faster_rcnn.da_head.da_roi_head import DAROIHeads


def add_swdarcnn_config(cfg):
    from detectron2.config import CfgNode as CN
    _C = cfg
    _C.MODEL.DA_HEADS = CN()
    
    _C.MODEL.DA_HEADS.LOCAL_ALIGNMENT_ON = True
    _C.MODEL.DA_HEADS.GLOBAL_ALIGNMENT_ON = True
    _C.MODEL.DA_HEADS.GAMMA = 5.0
    _C.MODEL.ROI_HEADS.CONTEXT_REGULARIZATION_ON = True
    _C.MODEL.ROI_HEADS.CONTEXT_REGULARIZATION_FEATURES = ['local_head_feature', 'global_head_feature']
    _C.DATASETS.SOURCE_DOMAIN = CN()
    _C.DATASETS.SOURCE_DOMAIN.TRAIN = ()
    _C.DATASETS.TARGET_DOMAIN = CN()
    _C.DATASETS.TARGET_DOMAIN.TRAIN = ()

    _C_ = _C.clone()
    _C.FEWSHOT_TUNING = CN()
    _C.FEWSHOT_TUNING.SOLVER = _C_.SOLVER # can not copy directly, because node is same, right one will be modified
    _C.FEWSHOT_TUNING.DATASETS = _C_.DATASETS
    _C.FEWSHOT_TUNING.BACKBONE_FROZEN = True
    _C.FEWSHOT_TUNING.DOMAIN_ADAPTATION_ON = True
    _C.FEWSHOT_TUNING.MODEL = CN()
    _C.FEWSHOT_TUNING.MODEL.WEIGHTS = ''
    _C.FEWSHOT_TUNING.TEST = _C_.TEST
    _C.FEWSHOT_TUNING.TEST.EVAL_PERIOD = 1000

def setup(args):
    cfg = get_cfg()
    add_swdarcnn_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    now = datetime.now()
    if args.tuning_only:
        cfg.OUTPUT_DIR = './outputs/output-tuning-{}'.format(now.strftime("%y-%m-%d_%H-%M"))
        if args.setting_token:
            cfg.OUTPUT_DIR = './outputs/output-tuning-{}-{}'.format(args.setting_token ,now.strftime("%y-%m-%d_%H-%M"))
    elif not args.resume:
        cfg.OUTPUT_DIR = './outputs/output-{}'.format(now.strftime("%y-%m-%d_%H-%M"))
        if args.setting_token:
            cfg.OUTPUT_DIR = './outputs/output-{}-{}'.format(args.setting_token ,now.strftime("%y-%m-%d_%H-%M"))
    cfg.freeze()
    if not args.test_images:
        default_setup(cfg, args)
    return cfg

def test_images(cfg):
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.data import MetadataCatalog
    from detectron2.data.datasets import load_voc_instances
    import cv2
    from pathlib import Path
    predictor = DefaultPredictor(cfg)

    for dataset_name in cfg.DATASETS.TEST:
        now = datetime.now()
        output_dir = Path(__file__).parent.parent/ 'test_images'/ (dataset_name + '-' + now.strftime("%y-%m-%d_%H-%M"))
        if not output_dir.parent.is_dir():
            output_dir.parent.mkdir()
        if not output_dir.is_dir():
            output_dir.mkdir()
        dirname = MetadataCatalog.get(dataset_name).get('dirname')
        split = MetadataCatalog.get(dataset_name).get('split')
        thing_classes = MetadataCatalog.get(dataset_name).get('thing_classes')
        for d in iter(load_voc_instances(dirname, split, thing_classes)):
            im = cv2.imread(d.get('file_name'))
            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(dataset_name), scale=1.2, instance_mode=ColorMode.IMAGE_BW)
            outputs = predictor(im)
            cv2.imwrite(str(output_dir/'{}.jpg').format(Path(d.get('file_name')).stem), v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()[:, :, ::-1])

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
    if args.test_images:
        test_images(cfg)
        return

    if args.tuning_only:
        assert cfg.FEWSHOT_TUNING.MODEL.WEIGHTS, 'FEWSHOT_TUNING.MODEL.WEIGHTS is needed'
        assert os.path.isfile(cfg.FEWSHOT_TUNING.MODEL.WEIGHTS), '{} not found'.format(cfg.FEWSHOT_TUNING.MODEL.WEIGHTS)
        trainer = FewShotTuner(cfg)
        trainer.resume_or_load(resume=args.resume)
        if not cfg.FEWSHOT_TUNING.DOMAIN_ADAPTATION_ON:
            FewShotTuner.freeze_da_heads(trainer)
        if cfg.FEWSHOT_TUNING.BACKBONE_FROZEN:
            FewShotTuner.freeze_backbone(trainer)
        return trainer.train()

    trainer = DATrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--tuning-only", action="store_true", help="perform few-shot tuning only")
    parser.add_argument("--test-images", action="store_true", help="output predicted bbox to test images")
    parser.add_argument("--setting-token", help="add some simple profile about this experiment, and add it to output directory name")
    args = parser.parse_args()
    print("Command Line Args:", args)


    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
