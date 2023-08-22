import glob
import json
import os
import random
from typing import List

import numpy as np
import torch
from tqdm import tqdm
import cv2

from mmhuman3d.core.cameras import build_cameras
# from mmhuman3d.core.conventions.keypoints_mapping import smplx
from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_idx,
    get_keypoint_idxs_by_part,
)
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.models.body_models.builder import build_body_model
from mmhuman3d.models.body_models.utils import transform_to_camera_frame
# from mmhuman3d.utils.transforms import aa_to_rotmat, rotmat_to_aa
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS

import pdb


@DATA_CONVERTERS.register_module()
class Pw3dNeuralConverter(BaseModeConverter):
    """3D Poses in the Wild dataset `Recovering Accurate 3D Human Pose in The
    Wild Using IMUs and a Moving Camera' ECCV'2018 More details can be found in
    the `paper.

    <https://virtualhumans.mpi-inf.mpg.de/papers/vonmarcardECCV18/
    vonmarcardECCV18.pdf>`__ .

    Args:
        modes (list): 'test' and/or 'train' for accepted modes
    """

    ACCEPTED_MODES = ['train', 'test', 'val']

    def __init__(self, modes: List = []) -> None:

        self.device = torch.device('cuda:0')
        self.misc = dict(
            bbox_source='by_dataset',
            cam_param_type='prespective',
            cam_param_source='original',
            smplx_source='neural_annot',
        )
        self.smplx_shape = {
            'betas': (-1, 10),
            'transl': (-1, 3),
            'global_orient': (-1, 3),
            'body_pose': (-1, 21, 3),
            # 'left_hand_pose': (-1, 15, 3),
            # 'right_hand_pose': (-1, 15, 3),
            # 'leye_pose': (-1, 3),
            # 'reye_pose': (-1, 3),
            # 'jaw_pose': (-1, 3),
            # 'expression': (-1, 10)
        }
        self.anno_key_map = {
            'shape': 'betas',
            'trans': 'transl',
            'root_pose': 'global_orient',
            'body_pose': 'body_pose', 
        }
        super(Pw3dNeuralConverter, self).__init__(modes)

    def convert_by_mode(self,
                        dataset_path: str,
                        out_path: str,
                        mode: str,
                        enable_multi_human_data: bool = False) -> dict:
        """
        Args:
            dataset_path (str): Path to directory where raw images and
            annotations are stored.
            out_path (str): Path to directory to save preprocessed npz file
            mode (str): Mode in accepted modes

        Returns:
            dict:
                A dict containing keys image_path, bbox_xywh, smpl, meta
                stored in HumanData() format
        """
        # use HumanData to store all data
        human_data = HumanData()

        # initialize output for human_data
        smplx_, smplx_extra_ = {}, {}
        for key in self.smplx_shape.keys():
            smplx_[key] = []
        keypoints2d_smplx_, keypoints3d_smplx_, = [], []
        keypoints2d_orig_ = []
        bboxs_ = {}
        for bbox_name in ['bbox_xywh']:
            bboxs_[bbox_name] = []
        meta_ = {}
        for key in ['focal_length', 'principal_point', 'height', 'width']:
            meta_[key] = []
        image_path_ = []

        # load train val test split
        anno_p = os.path.join(dataset_path, f'3DPW_{mode}_reformat.json')
        with open(anno_p, 'r') as f:
            info_annos = json.load(f)

        # load smplx annotaion
        smplx_p = os.path.join(dataset_path, f'3DPW_{mode}_SMPLX_NeuralAnnot.json')
        with open(smplx_p, 'r') as f:
            smplx_annos = json.load(f)

        # verify valid image and smplx
        print('Selecting valid image and smplx instances...')
        smplx_instances = list(smplx_annos.keys())
        for sid in smplx_instances:
            if sid not in info_annos.keys():
                smplx_annos.pop(sid)
        targeted_frame_ids = list(smplx_annos.keys())
        
        # init seed and size
        seed, size = '230821', '99999'
        size_i = min(int(size), len(targeted_frame_ids))
        random.seed(int(seed))
        targeted_frame_ids = targeted_frame_ids[:size_i]

        # init smplx model
        smplx_model = build_body_model(
            dict(
                type='SMPLX',
                keypoint_src='smplx',
                keypoint_dst='smplx',
                model_path='data/body_models/smplx',
                gender='neutral',
                num_betas=10,
                use_face_contour=True,
                flat_hand_mean=False,
                use_pca=False,
                batch_size=1)).to(self.device)

        print('Converting...')
        for sid in tqdm(targeted_frame_ids):

            smplx_anno = smplx_annos[sid]
            camera_param = info_annos[sid]['cam_param']
            info_anno = info_annos[sid]

            # get bbox
            width, height = info_anno['width'], info_anno['height']
            bbox_xywh = info_anno['bbox']
            if bbox_xywh[2] * bbox_xywh[3] > 0:
                bbox_xywh.append(1)
            else:
                bbox_xywh.append(0)

            # get image path
            imgp = os.path.join(dataset_path, 'imageFiles', str(info_anno['image_name']))
            if not os.path.exists(imgp):
                pdb.set_trace()
                print('missing image: ', imgp)
                continue
            image_path = imgp.replace(f'{dataset_path}{os.path.sep}', '')

            # get camera parameters and create camera
            focal_length = camera_param['focal']
            principal_point = camera_param['princpt']
            camera = build_cameras(
                dict(
                    type='PerspectiveCameras',
                    convention='opencv',
                    in_ndc=False,
                    focal_length=focal_length,
                    image_size=(width, height),
                    principal_point=principal_point)).to(self.device)

            # reformat smplx_anno
            smplx_param = {}
            for key in self.anno_key_map.keys():
                smplx_key = self.anno_key_map[key]
                smplx_shape = self.smplx_shape[smplx_key]
                smplx_param[smplx_key] = np.array(smplx_anno[key]).reshape(smplx_shape)
            
            # build smplx model and get output
            intersect_keys = list(
                set(smplx_param.keys()) & set(self.smplx_shape.keys()))
            body_model_param_tensor = {
                key: torch.tensor(
                    np.array(smplx_param[key]).reshape(self.smplx_shape[key]),
                    device=self.device, dtype=torch.float32)
                for key in intersect_keys}
            output = smplx_model(**body_model_param_tensor, return_joints=True)

            # get kps2d and 3d
            keypoints_3d = output['joints']
            keypoints_2d_xyd = camera.transform_points_screen(keypoints_3d)
            keypoints_2d = keypoints_2d_xyd[..., :2].detach().cpu().numpy()
            keypoints_3d = keypoints_3d.detach().cpu().numpy()

            # get kps2d original
            j2d_body = np.array(info_anno['openpose_result'])
            # j2d_ft = info_anno['foot_kpts']
            # j2d_lh = info_anno['lefthand_kpts']
            # j2d_rh = info_anno['righthand_kpts']
            # j2d_face = info_anno['face_kpts']

            # j2d = np.concatenate([j2d_body, j2d_ft, j2d_lh, j2d_rh, j2d_face], axis=0)
            j2d = j2d_body.reshape(-1, 3)

            # change conf 0, 1, 2 to 0, 1
            j2d_conf = j2d[:, -1]
            j2d_conf = (j2d_conf != 0).astype(int)
            j2d[:, -1] = j2d_conf

            # print('j2d_body', len(j2d_body))
            # print('j2d_lh', len(j2d_lh))
            # print('j2d_rh', len(j2d_rh))
            # print('j2d_face', len(j2d_face))

            # append image path
            image_path_.append(image_path)

            # append keypoints2d and 3d
            keypoints2d_smplx_.append(keypoints_2d)
            keypoints3d_smplx_.append(keypoints_3d)
            keypoints2d_orig_.append(j2d)

            # append bbox
            bboxs_['bbox_xywh'].append(bbox_xywh)

            # append smpl
            for key in smplx_param.keys():
                smplx_[key].append(smplx_param[key])

            # append meta
            meta_['principal_point'].append(principal_point)
            meta_['focal_length'].append(focal_length)
            meta_['height'].append(height)
            meta_['width'].append(width)

        # meta
        human_data['meta'] = meta_

        # image path
        human_data['image_path'] = image_path_

        # save bbox
        for bbox_name in bboxs_.keys():
            bbox_ = np.array(bboxs_[bbox_name]).reshape(-1, 5)
            human_data[bbox_name] = bbox_

        # save smplx
        # human_data.skip_keys_check = ['smplx']
        for key in smplx_.keys():
            smplx_[key] = np.concatenate(
                smplx_[key], axis=0).reshape(self.smplx_shape[key])
        for key in smplx_extra_.keys():
            smplx_[key] = np.array(smplx_extra_[key])
        human_data['smplx'] = smplx_

        # keypoints2d_smplx
        keypoints2d_smplx = np.concatenate(
            keypoints2d_smplx_, axis=0).reshape(-1, 144, 2)
        keypoints2d_smplx_conf = np.ones([keypoints2d_smplx.shape[0], 144, 1])
        keypoints2d_smplx = np.concatenate(
            [keypoints2d_smplx, keypoints2d_smplx_conf], axis=-1)
        keypoints2d_smplx, keypoints2d_smplx_mask = \
                convert_kps(keypoints2d_smplx, src='smplx', dst='human_data')
        human_data['keypoints2d_smplx'] = keypoints2d_smplx
        human_data['keypoints2d_smplx_mask'] = keypoints2d_smplx_mask

        # keypoints3d_smplx
        keypoints3d_smplx = np.concatenate(
            keypoints3d_smplx_, axis=0).reshape(-1, 144, 3)
        keypoints3d_smplx_conf = np.ones([keypoints3d_smplx.shape[0], 144, 1])
        keypoints3d_smplx = np.concatenate(
            [keypoints3d_smplx, keypoints3d_smplx_conf], axis=-1)
        keypoints3d_smplx, keypoints3d_smplx_mask = \
                convert_kps(keypoints3d_smplx, src='smplx', dst='human_data')
        human_data['keypoints3d_smplx'] = keypoints3d_smplx
        human_data['keypoints3d_smplx_mask'] = keypoints3d_smplx_mask

        # keypoints2d_orig
        keypoints2d_orig = np.concatenate(
            keypoints2d_orig_, axis=0).reshape(-1, 18, 3)
        # keypoints2d_orig_conf = np.ones([keypoints2d_orig.shape[0], 18, 1])
        # keypoints2d_orig = np.concatenate( 
        #     [keypoints2d_orig[:, :, :2], keypoints2d_orig_conf], axis=-1)
        keypoints2d_orig, keypoints2d_orig_mask = \
                convert_kps(keypoints2d_orig, src='pw3d', dst='human_data')
        human_data['keypoints2d_original'] = keypoints2d_orig
        human_data['keypoints2d_original_mask'] = keypoints2d_orig_mask

        # misc
        human_data['misc'] = self.misc
        human_data['config'] = f'pw3d_neural_annot_{mode}'

        # save
        human_data.compress_keypoints_by_mask()
        os.makedirs(out_path, exist_ok=True)
        out_file = os.path.join(
            out_path,
            f'pw3d_neural_{mode}_{seed}_{"{:05d}".format(size_i)}.npz')
        human_data.dump(out_file)




