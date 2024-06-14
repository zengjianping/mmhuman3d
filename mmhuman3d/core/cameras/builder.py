from mmengine.registry import Registry

CAMERAS = Registry('cameras')


def build_cameras(cfg):
    """Build cameras."""
    return CAMERAS.build(cfg)
