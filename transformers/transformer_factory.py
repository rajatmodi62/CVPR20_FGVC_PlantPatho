from transformers.utils import (
    DefaultTransformer, ImageTransformer, PolicyTransformer)


class TransformerFactory:
    def __init__(self):
        pass

    def get_transformer(self, height, width, pipe_type=None, policy=None):
        if width == None or height == None:
            print("[ Resize dims missing ]")
            exit()

        if isinstance(policy, list):
            return PolicyTransformer(height, width, policy)

        if pipe_type == "image":
            return ImageTransformer(height, width)
        elif pipe_type == "policy":
            return PolicyTransformer(height, width)
        else:
            return DefaultTransformer(height, width)
