python tools/train_net.py --config-file "/media/t2-503-c/Data4/ClayYou/detectron2-DA/configs/faster-rcnn_R_50_FPN_1x.yaml" --num-gpus 1 \
VIS_PERIOD 5000
# python tools/train_net.py --config-file "/media/t2-503-c/Data4/ClayYou/detectron2-DA/configs/swda_rcnn_clg_R_50_FPN_1x.yaml" --num-gpus 1 \
# VIS_PERIOD 5000 MODEL.DA_HEADS.GAMMA 5.0
# python tools/train_net.py --config-file "/media/t2-503-c/Data4/ClayYou/detectron2-DA/configs/swda_rcnn_clg_R_50_FPN_1x.yaml" --num-gpus 1 \
# --resume VIS_PERIOD 1 OUTPUT_DIR "outputs/output-21-09-11_20-30/"