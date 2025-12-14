class SculptHook:
    def on_part_fitted(self, **kwargs):
        """Called after a part is fitted (before adding to blender)."""
        pass

    def on_model_end(self, **kwargs):
        """Called after a model finished meshing."""
        pass
