# utils/visualization/base.py
from .objects import VisObject, PointCloudObject, MeshObject, ImageObject

class BaseRenderer:
    """
    所有渲染器的基类。
    """
    def render(self, obj: VisObject, **kwargs):
        """渲染单个对象"""
        raise NotImplementedError

    def render_multiple(self, objects, **kwargs):
        """渲染多个对象（场景）"""
        for o in objects:
            self.render(o, **kwargs)
