import argparse
import os

from mmhuman3d.data.data_converters import build_data_converter

DATASET_CONFIGS = dict(
    amass=dict(type='AmassConverter', prefix='AMASS_file'),
    coco=dict(type='CocoConverter'),
    coco_wholebody=dict(
        type='CocoWholebodyConverter', modes=['train', 'val'], prefix='coco'),
    crowdpose=dict(
        type='CrowdposeConverter', modes=['train', 'val', 'test', 'trainval']),
    pw3d=dict(type='Pw3dConverter', modes=['train', 'test']),
    mpii=dict(type='MpiiConverter'),
    h36m_p1=dict(
        type='H36mConverter',
        modes=['train', 'valid'],
        protocol=1,
        mosh_dir='data/datasets/h36m_mosh',
        prefix='h36m'),
    h36m_p2=dict(
        type='H36mConverter', modes=['valid'], protocol=2, prefix='h36m'),
    mpi_inf_3dhp=dict(type='MpiInf3dhpConverter', modes=['train', 'test']),
    penn_action=dict(type='PennActionConverter'),
    lsp_original=dict(type='LspConverter', modes=['train'], prefix='lsp'),
    lsp_dataset=dict(type='LspConverter', modes=['test']),
    lsp_extended=dict(type='LspExtendedConverter', prefix='lspet'),
    up3d=dict(
        type='Up3dConverter', modes=['trainval', 'test'], prefix='up-3d'),
    posetrack=dict(type='PosetrackConverter', modes=['train', 'val']),
    instavariety_vibe=dict(type='InstaVibeConverter', prefix='vibe_data'),
    eft=dict(
        type='EftConverter', modes=['coco_all', 'coco_part', 'mpii', 'lspet']),
    coco_hybrik=dict(type='CocoHybrIKConverter', prefix='coco/train_2017'),
    pw3d_hybrik=dict(type='Pw3dHybrIKConverter', prefix='hybrik_data'),
    h36m_hybrik=dict(
        type='H36mHybrIKConverter',
        modes=['train', 'test'],
        prefix='hybrik_data'),
    mpi_inf_3dhp_hybrik=dict(
        type='MpiInf3dhpHybrIKConverter',
        modes=['train', 'test'],
        prefix='hybrik_data'),
    surreal=dict(
        type='SurrealConverter', modes=['train', 'val', 'test'], run=0),
    spin=dict(
        type='SpinConverter',
        modes=['coco_2014', 'lsp', 'mpii', 'mpi_inf_3dhp', 'lspet'],
        prefix='spin_data'),
    vibe=dict(
        type='VibeConverter',
        modes=['pw3d', 'mpi_inf_3dhp'],
        pretrained_ckpt='data/checkpoints/spin.pth',
        prefix='vibe_data'),
    gta_human=dict(type='GTAHumanConverter', prefix='gta_human'),
    humman=dict(
        type='HuMManConverter', modes=['train', 'test'], prefix='humman'),
    cliff=dict(type='CliffConverter', modes=['coco', 'mpii']),

    # -------------- below datasets are done by WC --------------

    # -------------- multi-human dataset --------------
    agora=dict(
        type='AgoraConverter',  # synthetic
        prefix='agora',
        modes=[
            'validation_3840', 'train_3840', 'train_1280', 'validation_1280'
        ]),
    egobody=dict(
        type='EgobodyConverter',  # real, egocentric: single
        prefix='egobody',
        modes=[
            'egocentric_train', 'egocentric_test', 'egocentric_val',
            'kinect_train', 'kinect_test', 'kinect_val'
        ]),
    gta_human2=dict(
        type='GTAHuman2Converter',  # synthetic
        prefix='gta_human2',
        modes=['single', 'multiple']),
    synbody=dict(
        type='SynbodyConverter',  # synthetic
        prefix='synbody',
        modes=['train']),
    ubody=dict(
        type='UbodyConverter',  # real, has some single
        prefix='ubody',
        modes=['inter', 'intra']),

    # -------------- single-human dataset --------------
    behave=dict(
        type='BehaveConverter',  # real
        prefix='behave',
        modes=['train', 'test']),
    cimi4d=dict(
        type='Cimi4dConverter',  # real
        prefix='cimi4d',
        modes=['train']),
    moyo=dict(
        type='MoyoConverter',  # real
        prefix='moyo',
        modes=['train', 'val']),
    renbody=dict(
        type='RenbodyConverter',  # real
        prefix='renbody',
        modes=['train', 'train_highrescam', 'test', 'test_highrescam']),
    sgnify=dict(
        type='SgnifyConverter',  # real
        prefix='sgnify',
        modes=['train']),
    sloper4d=dict(
        type='Sloper4dConverter',  # real
        prefix='sloper4d',
        modes=['train']),
    sminchisescu=dict(
        type='ImarDatasetsConverter',  # real, 3 studio datasets
        prefix='sminchisescu-research-datasets',
        modes=['FIT3D', 'CHI3D', 'HumanSC3D']),
    ssp3d=dict(
        type='Ssp3dConverter',  # real
        prefix='ssp-3d'),

    # -------------- hand and face dataset (no complete smplx) --------------
    freihand=dict(  # in progress
        type='FreihandConverter',  # real, hand only
        prefix='freihand',
        modes=['train', 'val', 'test']),
    hanco=dict(  # in progress
        type='HancoConverter',  # real, hand only, arugmented background
        prefix='hanco',
        modes=['train', 'val']),
    interhand26m=dict(  # in progress
        type='Interhand26MConverter',  # real, hand only
        prefix='interhand26m',
        modes=['train', 'val', 'test']),

    # -------------- other dataset (no complete smpl/smplx) --------------
    humanart=dict(
        type='HumanartConverter',  # real, but have some human-like instances
        prefix='humanart',
        modes=['real_human', '2D_virtual_human', '3D_virtual_human']),
    shapy=dict(
        type='ShapyConverter',  # real
        prefix='shapy',
        modes=['test', 'val']),
)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert datasets')

    parser.add_argument(
        '--root_path',
        type=str,
        required=True,
        help='the root path of original data')

    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='the path to store the preprocessed npz files')

    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        required=True,
        default=[],
        help=f'Supported datasets: {list(DATASET_CONFIGS.keys())}')

    parser.add_argument(
        '--modes',
        type=str,
        nargs='+',
        required=False,
        default=[],
        help='Need to comply with supported modes'
        'specified in tools/convert_datasets.py')

    parser.add_argument(
        '--prefix',
        type=str,
        required=False,
        default=None,
        help='If you want to specify you folder name, please use this')

    parser.add_argument(
        '--enable_multi_human_data',
        type=bool,
        default=False,
        help='Whether to generate a multi-human data')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    datasets = (
        DATASET_CONFIGS.keys() if args.datasets == ['all'] else args.datasets)

    for dataset in datasets:
        print(f'[{dataset}] Converting ...')
        cfg = DATASET_CONFIGS[dataset]

        if ('modes' in cfg.keys()) and (args.modes != []):
            assert all(x in cfg['modes'] for x in args.modes), \
                f'Unsupported mode found, supported mode for' \
                f'{cfg["prefix"]} is {cfg["modes"]}'
            cfg['modes'] = args.modes
        elif ('modes' in cfg.keys()) and (args.modes == []):
            print(f'For {cfg["prefix"]}, modes: {cfg["modes"]} are available,'
                  ' process all modes as not specified')
            args.modes = cfg['modes']

        if args.prefix is not None:
            prefix = args.prefix
        else:
            prefix = cfg.pop('prefix', dataset)
        input_path = os.path.join(args.root_path, prefix)
        data_converter = build_data_converter(cfg)
        data_converter.convert(input_path, args.output_path)
        print(f'[{dataset}] Converting finished!')


if __name__ == '__main__':
    main()
