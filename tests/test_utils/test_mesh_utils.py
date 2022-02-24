import pytest
from pytorch3d.utils import torus

from mmhuman3d.utils.mesh_utils import (
    mesh_to_pointcloud_vc,
    save_meshes_as_objs,
    save_meshes_as_plys,
)


def test_save_meshes():
    Torus = torus(r=10, R=20, sides=100, rings=100)
    # wrong files
    with pytest.raises(AssertionError):
        save_meshes_as_plys(meshes=Torus)

    # No input
    with pytest.raises(AssertionError):
        save_meshes_as_plys()

    # File suffix wrong
    with pytest.raises(AssertionError):
        save_meshes_as_plys(Torus, files=['1.obj'])

    save_meshes_as_plys(Torus, files=['1.ply'])
    save_meshes_as_plys(
        verts=Torus.verts_padded(), faces=Torus.faces_padded(), files='1.ply')
    save_meshes_as_plys(
        Torus,
        verts=Torus.verts_padded(),
        faces=Torus.faces_packed(),
        files='1.ply')
    save_meshes_as_objs(Torus, files='1.obj')


def test_mesh2pointcloud():
    Torus = torus(r=10, R=20, sides=100, rings=100)
    Torus.textures = None
    with pytest.raises(AssertionError):
        mesh_to_pointcloud_vc(Torus)
