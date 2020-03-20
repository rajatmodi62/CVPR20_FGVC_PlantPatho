from torch.nn import (CrossEntropyLoss)
from losses.focal_loss import FocalLoss

class LossFactory:
    def __init__(self):
        pass

    def get_loss_function( self, function_name ):
        loss_function = None
        
        if function_name == 'focal-loss':
            loss_function = FocalLoss()
        if function_name == 'cross-entropy-loss':
            loss_function = CrossEntropyLoss()
        
        return loss_function