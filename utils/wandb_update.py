import wandb
from utils.print_util import cprint


def wandb_init(config):
    wandb.init(project="plantpatho-2020")

    wandb.config.experiment_name = config['experiment_name']
    wandb.config.seed = config['seed']
    wandb.config.model = config['model']['name']
    wandb.config.prediction_type = config['model']['pred_type']
    wandb.config.optimiser = config['optimiser']['name']
    wandb.config.learning_rate = config['optimiser']['hyper_params']['learning_rate']
    wandb.config.loss_function = config['loss_function']['name']
    wandb.config.resize_dims = config['train_dataset']['resize_dims']
    wandb.config.epochs = config["epochs"]
    wandb.config.batch_size = config["batch_size"]

    cprint("[ Setting up project on W&B ]", type="info3")
    return True


def publish_intermediate(results, best_val_loss, best_kaggle_metric):
    wandb.config.update(
        {
            "best_val_loss": best_val_loss,
            "best_kaggle_metric": best_kaggle_metric
        }, 
        allow_val_change=True
    )
    return wandb.log(results)
