python tools/train_net.py --config-file "/media/t2-503-c/Data4/ClayYou/detectron2-DA/configs/swda_rcnn_clg_R_50_FPN_1x.yaml" --num-gpus 1 \
--tuning-only --setting-token "all-class-org-sn" FEWSHOT_TUNING.MODEL.WEIGHTS "outputs/output-21-09-15_18-56/model_0089999.pth" VIS_PERIOD 5000
