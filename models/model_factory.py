from torch import nn
from models.efficientnet import EfficientNet
import torchvision.models as models


class ModelFactory():
    def __init__(self):
        pass

    def get_model(self, model_name, model_params=None, num_classes=4, tuning_type='feature_extraction'):
        model = None
        if model_name == 'efficientnet-b7':
            print("[ Model Efficientnet B7 loaded ]")
            model = EfficientNet.from_pretrained(num_classes=num_classes)
        if model_name == 'densenet-161':
            print("[ Model Densenet 161 loaded ]")
            model = models.densenet161(pretrained=True)
            if tuning_type == 'feature_extraction':
                for param in model.parameters():
                    param.requires_grad = False
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)
        return model
