# Copyright (c) OpenMMLab. All rights reserved.

from mmengine.registry import Registry

from .temporal_encoder import TemporalGRUEncoder

NECKS = Registry('necks')

NECKS.register_module(name='TemporalGRUEncoder', module=TemporalGRUEncoder)


def build_neck(cfg):
    """Build neck."""
    if cfg is None:
        return None
    return NECKS.build(cfg)
