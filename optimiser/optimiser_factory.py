from torch.optim import (RMSprop, Adam, AdamW)


class OptimiserFactory:
    def __init__(self):
        pass

    def get_optimiser(self, params, optimiser_name, hyper_params):
        if optimiser_name == 'RMSprop':
            print("[ Optimiser : RMSprop ]")
            return RMSprop(
                params,
                lr=hyper_params['learning_rate'],
                weight_decay=hyper_params['weight_decay'],
                momentum=hyper_params['momentum'],
                centered=hyper_params['centered']
            )
        if optimiser_name == 'Adam':
            print("[ Optimiser : Adam ]")
            return Adam(
                params,
                lr=hyper_params['learning_rate'],
                weight_decay = hyper_params['weight_decay']
            )
        if optimiser_name == 'AdamW':
            print("[ Optimiser : AdamW ]")
            return AdamW(
                params,
                lr=hyper_params['learning_rate']
                weight_decay=hyper_params['weight_decay']
            )
