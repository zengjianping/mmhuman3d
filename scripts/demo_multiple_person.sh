#!/bin/bash

python demo/estimate_smpl.py \
    configs/hmr/resnet50_hmr_pw3d.py \
    data/checkpoints/resnet50_hmr_pw3d-04f40f58_20211201.pth \
    --multi_person_demo \
    --tracking_config demo/mmtracking_cfg/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py \
    --input_path  demo/resources/multi_person_demo.mp4 \
    --show_path vis_results/multi_person_demo.mp4 \
    --smooth_type savgol \
    --speed_up_type deciwatch \
    --draw_bbox
