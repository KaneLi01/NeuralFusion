# utils/visualization/base.py
from ..objects import Object3D, PointCloudObject, MeshObject

class BaseRenderer:
    """
    所有渲染器的基类。
    """
    def render(self, obj: Object3D, **kwargs):
        """渲染单个对象"""
        raise NotImplementedError

    def render_multiple(self, objects, **kwargs):
        """渲染多个对象（场景）"""
        for o in objects:
            self.render(o, **kwargs)
