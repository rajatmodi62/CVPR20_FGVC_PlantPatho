from models.efficientnet import EfficientNet
import torchvision.models as models

class ModelFactory():
    def __init__(self):
        pass

    def get_model( self, model_name, model_params=None ):
        model = None
        if model_name == 'efficientnet-b7':
            print("[ Model Efficientnet B7 loaded! ]")
            model = EfficientNet.from_pretrained()
        if model_name == 'densenet-161':
            print("[ Model Densenet 161 loaded! ]")
            model = models.densenet161(pretrained=True)
        return model