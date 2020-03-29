from torch.optim.lr_scheduler import (StepLR, CosineAnnealingLR)


class SchedulerFactory:
    def __init__(self):
        pass

    def get_scheduler(self, optimiser, name, hyper_params):
        scheduler = None
        if name == 'StepLR':
            print("[ Scheduler : StepLR ]")
            scheduler = StepLR(
                optimiser, 
                hyper_params['step'], 
                hyper_params['lr_decay']
            )
        elif name == 'CosineAnnealingLR':
            print("[ Scheduler : CosineAnnealingLR ]")
            scheduler = CosineAnnealingLR(
                optimiser,
                hyper_params['T_max'],
                hyper_params['eta_min'],
                hyper_params['last_epoch']
            )

        return scheduler
