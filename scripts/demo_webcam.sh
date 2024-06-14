#!/bin/bash

python demo/webcam_demo.py \
    --synchronous true \
    --mesh_reg_config configs/pare/hrnet_w32_conv_pare_coco.py \
    --mesh_reg_checkpoint data/checkpoints/hrnet_w32_conv_pare_mosh.pth \
    --det_config demo/mmdetection_cfg/ssdlite_mobilenetv2_scratch_600e_onehand.py \
    --cam-id demo/resources/test_videos/2024-04-17_17-40-07_c03.mp4

