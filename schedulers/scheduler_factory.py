from torch.optim.lr_scheduler import (StepLR)


class SchedulerFactory:
    def __init__(self):
        pass

    def get_scheduler(self, optimiser, name, hyper_params):
        scheduler = None
        if name == 'StepLR':
            print("[ Scheduler : StepLR ]")
            scheduler = StepLR(
                optimiser, hyper_params['step'], hyper_params['lr_decay'])

        return scheduler
