from transformer.utils import (
    DefaultTransformer, ImageTransformer, ImageTTATransformer, PolicyTransformer)


class TransformerFactory:
    def __init__(self):
        pass

    def get_transformer(self, height, width, pipe_type=None, auto_aug_policy=None):
        if width == None or height == None:
            print("[ Resize dims missing ]")
            exit()

        if isinstance(auto_aug_policy, list):
            return PolicyTransformer(height, width, auto_aug_policy)

        if pipe_type == "image":
            return ImageTransformer(height, width)
        if pipe_type == "image_tta":
            return ImageTTATransformer(height, width)
        elif pipe_type == "policy":
            return PolicyTransformer(height, width)
        else:
            return DefaultTransformer(height, width)
