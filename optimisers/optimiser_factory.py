from torch.optim import (RMSprop, Adam)


class OptimiserFactory:
    def __init__(self):
        pass

    def get_optimiser(self, optimiser_name, params, hyper_param_dict):
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
