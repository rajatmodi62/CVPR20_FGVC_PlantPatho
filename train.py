import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from os import (path, environ)
from losses.loss_factory import LossFactory
from optimisers.optimiser_factory import OptimiserFactory
from schedulers.scheduler_factory import SchedulerFactory
from dataset.dataset_factory import DatasetFactory
from models.model_factory import ModelFactory
from transformers.transformer_factory import TransformerFactory
from utils.experiment_utils import ExperimentHelper
from utils.custom_bar import CustomBar
from utils.seed_backend import seed_all

# stop tensorboard warnings
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train(config, device, policy=None):
    # Create pipeline objects
    dataset_factory = DatasetFactory(org_data_dir='./data')

    transformer_factory = TransformerFactory()

    model_factory = ModelFactory()

    writer = SummaryWriter(
        log_dir=path.join(
            'runs', config['experiment_name']
        )
    )

    experiment_helper = ExperimentHelper(
        config['experiment_name'],
        config['validation_frequency'],
        tb_writer=writer,
        overwrite=True,
        publish=config['publish'],
        config=config
    )

    optimiser_factory = OptimiserFactory()

    loss_factory = LossFactory()

    scheduler_factory = SchedulerFactory()

    # ==================== Model training / validation setup ========================

    training_dataset = dataset_factory.get_dataset(
        'train',
        config['train_dataset']['name'],
        transformer_factory.get_transformer(
            height=config['train_dataset']['resize_dims'],
            width=config['train_dataset']['resize_dims'],
            pipe_type=config['train_dataset']['transform'],
            policy=policy
        ),
        config['train_dataset']['fold']
    )

    validation_dataset = dataset_factory.get_dataset(
        'val',
        config['val_dataset']['name'],
        transformer_factory.get_transformer(
            height=config['val_dataset']['resize_dims'],
            width=config['val_dataset']['resize_dims'],
            pipe_type=config['val_dataset']['transform']
        ),
        config['val_dataset']['fold']
    )

    model = model_factory.get_model(
        config['model']['name'],
        config['num_classes'],
        config['model']['pred_type'],
        config['model']['hyper_params'],
        config['model']['tuning_type'],
        config['model']['pre_trained_path'],
        config['model']['weight_type']
    ).to(device)

    optimiser = optimiser_factory.get_optimiser(
        model.parameters(),
        config['optimiser']['name'],
        config['optimiser']['hyper_params']
    )

    scheduler = None
    if config['scheduler']:
        scheduler = scheduler_factory.get_scheduler(
            optimiser,
            config['scheduler']['name'],
            config['scheduler']['hyper_params']
        )

    loss_function = loss_factory.get_loss_function(
        config['loss_function']['name'],
        config['model']['pred_type'],
        config['loss_function']['hyper_params']
    )

    batch_size = config["batch_size"]

    # ===============================================================================

    # =================== Model training / validation loop ==========================

    with CustomBar(config["epochs"], len(training_dataset), batch_size) as progress_bar:

        for i in range(config["epochs"]):
            # progress bar update
            progress_bar.update_epoch_info(i)

            # set model to training mode
            model.train()

            train_output_list = []
            train_target_list = []
            for batch_ndx, sample in enumerate(DataLoader(training_dataset, batch_size=batch_size)):
                # progress bar update
                progress_bar.update_batch_info(batch_ndx)

                input, target = sample
                input = input.to(device)
                target = target.to(device)
                input.requires_grad = False

                # flush accumulators
                optimiser.zero_grad()

                # forward pass
                output = model.forward(input)

                # loss calculation
                loss = loss_function(
                    output,
                    target
                )

                # backward pass
                loss.backward()

                # update
                optimiser.step()

                # update lr using scheduler
                if scheduler:
                    scheduler.step()

                if experiment_helper.should_trigger(i):
                    train_output_list.append(output)
                    train_target_list.append(target)

                # progress bar update
                progress_bar.step()

            # set model to evaluation mode
            model.eval()

            # Do a loss check on val set per epoch
            if experiment_helper.should_trigger(i):
                val_output_list = []
                val_target_list = []
                for batch_ndx, sample in enumerate(DataLoader(validation_dataset, batch_size=1)):
                    with torch.no_grad():
                        input, target = sample
                        input = input.to(device)
                        target = target.to(device)

                        output = model.forward(input)

                        val_output_list.append(output)
                        val_target_list.append(target)

                val_output_list = torch.cat(val_output_list, dim=0)
                val_target_list = torch.cat(val_target_list, dim=0)
                train_output_list = torch.cat(train_output_list, dim=0)
                train_target_list = torch.cat(train_target_list, dim=0)

                # validate model
                experiment_helper.validate(
                    config['model']['pred_type'],
                    config['num_classes'],
                    loss_function,
                    val_output_list,
                    val_target_list,
                    train_output_list,
                    train_target_list,
                    i
                )

                # save model weights
                if experiment_helper.is_progress():
                    experiment_helper.save_checkpoint(
                        model.state_dict()
                    )

    # publish final
    config['publish'] and experiment_helper.publish_final(config)

    return (experiment_helper.best_val_loss, experiment_helper.best_val_roc)

    # ===============================================================================
