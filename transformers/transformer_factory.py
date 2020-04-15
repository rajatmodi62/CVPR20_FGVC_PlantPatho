from transformers.utils import (DefaultTransformer, ImageTransformer)


class TransformerFactory:
    def __init__(self):
        pass

    def get_transformer(self, height, width, pipe_type="default"):
        if width == None or height == None:
            print("[ Resize dims missing ]")
            exit()

        if pipe_type == "default":
            return DefaultTransformer(height, width)
        elif pipe_type == "image":
            return ImageTransformer(height, width)
