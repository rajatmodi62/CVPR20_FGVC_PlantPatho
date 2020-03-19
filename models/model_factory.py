from models.efficientnet import EfficientNet

class ModelFactory():
    def __init__(self):
        pass

    def get_model( self, model_name, model_params=None ):
        model = None
        if model_name == 'efficientnet-b7':
            print("[ Model Efficientnet B7 loaded! ]")
            model = EfficientNet.from_pretrained()
        return model