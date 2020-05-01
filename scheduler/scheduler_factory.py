from torch.optim.lr_scheduler import (StepLR, CosineAnnealingLR)
from transformers import get_cosine_schedule_with_warmup


class SchedulerFactory:
    def __init__(self):
        pass

    def get_scheduler(self, optimiser, name, hyper_params, epochs, iter_per_epoch):
        scheduler = None
        if name == 'step-lr':
            print("[ Scheduler : Step LR ]")
            scheduler = StepLR(
                optimiser,
                hyper_params['step'],
                hyper_params['lr_decay']
            )
        elif name == 'cosineAnnealing-lr':
            print("[ Scheduler : Cosine Annealing LR ]")
            scheduler = CosineAnnealingLR(
                optimiser,
                T_max=epochs*iter_per_epoch
            )
        elif name == 'cosineAnnealing-warmup-lr':
            print("[ Scheduler : Cosine Annealing LR with Warmup ]")
            scheduler = get_cosine_schedule_with_warmup(
                optimiser,
                num_warmup_steps=iter_per_epoch * 5,
                num_training_steps=iter_per_epoch * epochs
            )

        return scheduler
