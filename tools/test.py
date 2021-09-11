import os
import sys
sys.path.append(os.getcwd())

from detectron2.modeling import build_backbone, build_model
from detectron2.config import get_cfg
import torch


from faster_rcnn.meta_archs.swda_rcnn import SWDARCNN
from faster_rcnn.backbones.fpn_da import build_resnet_fpn_da_backbone
from faster_rcnn.da_head.da_head import AlignmentHead
from faster_rcnn.da_head.da_roi_head import DAROIHeads
# cfg = get_cfg()
# cfg.merge_from_file('/media/t2-503-c/Data4/ClayYou/detectron2-DA/test/faster_rcnn_R_50_FPN_1x.yaml')


    

# print(cfg)
# network = build_backbone(cfg)
# x = {'res2':torch.zeros((1,256,100,100)),'res3':torch.zeros((1,256,100,100)),'res4':torch.zeros((1,256,100,100)),'res5':torch.zeros((1,256,100,100))}
# x = torch.zeros((1,3,640,640))

# print(network.forward(x).keys())




def add_swdarcnn_config(cfg):
    from detectron2.config import CfgNode as CN
    _C = cfg
    _C.DA_HEADS = CN()
    _C.DA_HEADS.DOMAIN_ADAPTATION_ON = True
    _C.DA_HEADS.LOCAL_ALIGNMENT_ON = True
    _C.DA_HEADS.GLOBAL_ALIGNMENT_ON = True
    _C.DA_HEADS.GAMMA = 5.0
    _C.MODEL.ROI_HEADS.CONTEXT_REGULARIZATION_ON = True
    _C.MODEL.ROI_HEADS.CONTEXT_REGULARIZATION_FEATURES = ['local_head_feature', 'global_head_feature']
    _C.DATASETS.SOURCE_DOMAIN = CN()
    _C.DATASETS.SOURCE_DOMAIN.TRAIN = ()
    _C.DATASETS.TARGET_DOMAIN = CN()
    _C.DATASETS.TARGET_DOMAIN.TRAIN = ()



def setup():
    cfg = get_cfg()
    add_swdarcnn_config(cfg)
    cfg.merge_from_file('/media/t2-503-c/Data4/ClayYou/detectron2-DA/configs/swda_rcnn_clg_R_50_FPN_1x.yaml')
    return cfg

from detectron2.data.datasets import register_pascal_voc
from detectron2.data import build_detection_train_loader
dataset_dir = '../Data/itri-taiwan-416-VOCdevkit2007/'
classes = ('person', 'two-wheels', 'four-wheels')
years = 2007
split = 'train' # "train", "test", "val", "trainval"
meta_name = 'itri-taiwan-416_{}'.format(split)
register_pascal_voc(meta_name, dataset_dir, split, years, classes)

def set_batchnorm_fix(m):
    classname = m.__class__.__name__

    print(classname, m.requires_grad)
    # if classname.find('BatchNorm') != -1:
    #     for p in m.parameters(): 
    #         p.requires_grad=False
# x = [{'image':torch.zeros((3,640,640))}]
cfg = setup()
cfg.DATASETS.TRAIN = (meta_name,)
# print(cfg)
dataloader = iter(build_detection_train_loader(cfg))
model = build_model(cfg)
x = next(dataloader)
# model.apply(set_batchnorm_fix)

model.da_heads.requires_grad = False
for p in model.da_heads.parameters():
    p.requires_grad = False

# print(model.da_heads.requires_grad)

# from detectron2.utils.events import EventStorage
# with EventStorage() as storage:
#     print(model(x, input_domain = 'target'))

