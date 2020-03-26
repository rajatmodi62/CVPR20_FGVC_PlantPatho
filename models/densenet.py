import torchvision.models as models
from torch import nn
from torch.nn import functional as F

class DenseNet( nn.Module ):
    def __init__(self, num_classes, pretrained=False):
        self.model = models.densenet161(pretrained=pretrained)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, num_classes)

    def forward( self, x ):
        out = self.model.forward( x )
        out = F.sigmoid( out )