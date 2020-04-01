from torch.optim import (RMSprop, Adam, AdamW)


class OptimiserFactory:
    def __init__(self):
        pass

    def get_optimiser(self, params, optimiser_name, hyper_param_dict):
        if optimiser_name == 'RMSprop':
            print("[ Optimiser : RMSprop ]")
            return RMSprop(
                params,
                lr=hyper_param_dict['learning_rate'],
                weight_decay=hyper_param_dict['weight_decay'],
                momentum=hyper_param_dict['momentum'],
                centered=hyper_param_dict['centered']
            )
        if optimiser_name == 'Adam':
            print("[ Optimiser : Adam ]")
            return Adam(
                params,
                lr=hyper_param_dict['learning_rate']
            )
        if optimiser_name == 'AdamW':
            print("[ Optimiser : AdamW ]")
            return AdamW(
                params,
                lr=hyper_param_dict['learning_rate']
            )
