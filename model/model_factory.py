import torch
from torch import nn
from os import path
import pretrainedmodels
import torchvision.models as models
from model.efficientnet import EfficientNet


class ModelFactory():
    def __init__(self):
        pass

    def get_model(self, model_name, num_classes, pred_type, hyper_params=None, tuning_type='feature-extraction', pre_trained_path=None, weight_type=None):
        if pred_type == 'regression':
            adjusted_num_classes = 1
        elif pred_type == 'mixed':
            adjusted_num_classes = num_classes + 1
        else:
            adjusted_num_classes = num_classes

        model = None

        if model_name == 'efficientnet-b7':
            print("[ Model : Efficientnet B7 ]")
            model = EfficientNet.from_pretrained(model_name='efficientnet-b7', advprop=True)
            if tuning_type == 'feature-extraction':
                for param in model.parameters():
                    param.requires_grad = False
            num_ftrs = model._fc.in_features
            # model._fc = nn.Sequential(
            #     # nn.Linear(num_ftrs, num_classes),
            #     # nn.Sigmoid(),
            #     nn.Linear(num_ftrs, adjusted_num_classes)
            # )
            model._fc = nn.Sequential(
                nn.Linear(num_ftrs, 1000, bias=True),
                nn.ReLU(),
                nn.Dropout(p=hyper_params['fc_drop_out']),
                nn.Linear(1000, adjusted_num_classes, bias=True)
            )

            if hyper_params is not None:
                model._bn_mom = hyper_params['batch_norm_momentum']

        if model_name == 'efficientnet-b4':
            print("[ Model : Efficientnet B4 ]")
            model = EfficientNet.from_pretrained(
                model_name='efficientnet-b4', advprop=True)
            if tuning_type == 'feature-extraction':
                for param in model.parameters():
                    param.requires_grad = False
            num_ftrs = model._fc.in_features
            model._fc = nn.Sequential(
                # nn.Linear(num_ftrs, num_classes),
                # nn.Sigmoid(),
                nn.Linear(num_ftrs, adjusted_num_classes)
            )

            if hyper_params is not None:
                model._bn_mom = hyper_params['batch_norm_momentum']

        if model_name == 'efficientnet-b5':
            print("[ Model : Efficientnet B5 ]")
            model = EfficientNet.from_pretrained(
                model_name='efficientnet-b5', advprop=False)
            if tuning_type == 'feature-extraction':
                for param in model.parameters():
                    param.requires_grad = False
            num_ftrs = model._fc.in_features
            model._fc = nn.Sequential(
                nn.Linear(num_ftrs, 1000, bias=True),
                nn.ReLU(),
                nn.Dropout(p=hyper_params['fc_drop_out']),
                nn.Linear(1000, adjusted_num_classes, bias=True)
            )

            # if hyper_params is not None:
            #     model._bn_mom = hyper_params['batch_norm_momentum']

        if model_name == 'densenet-161':
            print("[ Model : Densenet 161 ]")
            model = models.densenet161(pretrained=True)
            if tuning_type == 'feature-extraction':
                for param in model.parameters():
                    param.requires_grad = False
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Linear(num_ftrs, adjusted_num_classes)
            )

        if model_name == 'resnet34':
            print("[ Model : Resnet 34 ]")
            model = pretrainedmodels.__dict__[
                'resnet34'](pretrained='imagenet')

            model.avgpool = nn.AdaptiveAvgPool2d(1)
            in_features = model.last_linear.in_features
            model.last_linear = nn.Sequential(
                nn.Linear(in_features, adjusted_num_classes)
            )

        tuning_type and print("[ Tuning type : ", tuning_type, " ]")
        print("[ Prediction type : ", pred_type, " ]")

        # if model needs to resume from pretrained weights
        if pre_trained_path:
            weight_path = 'weights.pth'
            if weight_type == 'best_val_kaggle_metric':
                weight_path = 'weights_kaggle_metric.pth'
            elif weight_type == 'best_val_loss':
                weight_path = 'weights_loss.pth'
            weight_path = path.join(
                'results', pre_trained_path, weight_path)

            if path.exists(weight_path):
                print("[ Loading checkpoint : ",
                      pre_trained_path, " ]")
                model.load_state_dict(torch.load(
                    weight_path
                    # ,map_location={'cuda:1': 'cuda:0'}
                ))
            else:
                print("[ Provided pretrained weight path is invalid ]")
                exit()

            print("[ Weight type : ", weight_type if weight_type else "Last Epoch", " ]")

        return model
