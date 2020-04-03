from transformers.utils import (DefaultTransformer, ImageTransformer)


class TransformerFactory:
    def __init__(self):
        pass

    def get_transformer(self, height=2048, width=1365, pipe_type="default"):
        if pipe_type == "default":
            return DefaultTransformer(height, width)
        elif pipe_type == "image":
            return ImageTransformer(height, width)
