import wandb
import torch
from utils.print_util import cprint


def wandb_init(config):
    cprint("[ Setting up project on W&B ]", type="info3")

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

    # saving config files to W&B
    wandb.save('./config/' + config['experiment_name'] + '.yml')
    return True


def publish_intermediate(results, best_val_loss, best_kaggle_metric, output_list, target_list):
    wandb.run.summary["best_val_loss"] = best_val_loss
    wandb.run.summary["best_kaggle_metric"] = best_kaggle_metric

    # saving confusion matrix (image)
    wandb.sklearn.plot_confusion_matrix(
        torch.argmax(target_list, dim=1).numpy(),
        torch.argmax(output_list, dim=1).numpy(),
        ['H', 'MD', 'R', 'S']
    )

    return wandb.log(results)
