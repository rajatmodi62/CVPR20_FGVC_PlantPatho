import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from os import (path, environ)
from tqdm import (trange, tqdm)
from losses.loss_factory import LossFactory
from optimisers.optimiser_factory import OptimiserFactory
from schedulers.scheduler_factory import SchedulerFactory
from dataset.dataset_factory import DatasetFactory
from models.model_factory import ModelFactory
from transformers.transformer_factory import TransformerFactory
from utils.experiment_utils import ExperimentHelper

# stop tensorboard warnings
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train(config, device):
    # Create pipeline objects
    dataset_factory = DatasetFactory(org_data_dir='./data')

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
        overwrite=True
    )

    optimiser_factory = OptimiserFactory()

    loss_factory = LossFactory()

    scheduler_factory = SchedulerFactory()

    # ==================== Model training / validation setup ========================

    training_dataset = dataset_factory.get_dataset(
        'train',
        config['train_dataset']['name'],
        TransformerFactory(
            # pipe_type="image",
            height=config['train_dataset']['resize_dims'],
            width=config['train_dataset']['resize_dims'],
        ),
        config['train_dataset']['fold']
    )

    validation_dataset = dataset_factory.get_dataset(
        'val',
        config['val_dataset']['name'],
        TransformerFactory(
            height=config['val_dataset']['resize_dims'],
            width=config['val_dataset']['resize_dims'],
        ),
        config['val_dataset']['fold']
    )

    model = model_factory.get_model(
        config['model']['name'],
        config['num_classes'],
        config['model']['pred_type'],
        config['model']['hyper_params'],
        config['model']['tuning_type']
    ).to(device)

    if config['model']['pre_trained_path']:
        weight_path = path.join(
            'results', config['model']['pre_trained_path'], 'weights.pth')
        if path.exists(weight_path):
            print("[ Resuming traning using ", config['model']['pre_trained_path'], " ]")
            model.load_state_dict(torch.load(weight_path))
        else:
            print("[ Provided pretrained weight path is invalid ]")

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
        config['loss_function']['hyper_params']
    )

    batch_size = config["batch_size"]

    # ===============================================================================

    # =================== Model training / validation loop ==========================

    for i in trange(config["epochs"], desc="Epochs : "):

        # set model to training mode
        model.train()

        train_output_list = []
        train_target_list = []
        for batch_ndx, sample in enumerate(DataLoader(training_dataset, batch_size=batch_size)):
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

    # ===============================================================================
