python tools/train_net.py --config-file "/media/t2-503-c/Data4/ClayYou/detectron2-DA/configs/swda_rcnn_clg_R_50_FPN_1x.yaml" --num-gpus 1 \
--tuning-only FEWSHOT_TUNING.MODEL.WEIGHTS "outputs/output-21-09-13_18-17/model_0119999.pth" VIS_PERIOD 5000 MODEL.DA_HEADS.GAMMA 5.0
