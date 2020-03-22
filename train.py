import torch
from os import path
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from transformers.transformer_factory import TransformerFactory
from dataset.dataset_factory import DatasetFactory
from optimisers.optimiser_factory import OptimiserFactory
from models.model_factory import ModelFactory
from losses.loss_factory import LossFactory
from torch.utils.data import DataLoader
from utils.config_parser import (get_config_data)
from utils.experiment_utils import ExperimentHelper
from utils.check_gpu import get_training_device

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("experiment_file",
                    help="The name of the experiment config file")
args = parser.parse_args()

# Get experiment config values
if args.experiment_file is None:
    exit()
config = get_config_data(path.join('config', args.experiment_file))

# Setup tensorboard support
writer = SummaryWriter()

# Get GPU / CPU device instance
device = get_training_device()

# Create pipeline objects
optimiser_factory = OptimiserFactory()
dataset_factory = DatasetFactory(org_data_dir='./data')
model_factory = ModelFactory()
loss_factory = LossFactory()
experiment_helper = ExperimentHelper(
    config['experiment_name'], 
    True, 
    config['validation_frequency'],
    writer
)

# ===================== Model training / validation ==========================

# setup
training_dataset = dataset_factory.get_dataset(
    'train',
    config['dataset']['name'],
    TransformerFactory(
        height=config['dataset']['resize_dims'],
        width=config['dataset']['resize_dims'],
        pipe_type="image",
    ),
    config['dataset']['fold']
)

validation_dataset = dataset_factory.get_dataset(
    'val',
    config['dataset']['name'],
    TransformerFactory(
        height=config['dataset']['resize_dims'],
        width=config['dataset']['resize_dims'],
    ),
    config['dataset']['fold']
)

model = model_factory.get_model(
    config['model']['name']
).to(device)

optimiser = optimiser_factory.get_optimiser(
    config['optimiser']['name'], 
    model.parameters(), 
    config['optimiser']['hyper_params']
)

loss_function = loss_factory.get_loss_function(
    config['loss_function']['name']
)

# training / validation loop
batch_size = config["batch_size"]

for i in trange(config["epochs"], desc="Epochs : "):

    # set model to training mode
    model.train()

    train_output_list = []
    train_target_list = []
    for batch_ndx, sample in enumerate(DataLoader(training_dataset, batch_size=batch_size)):
        input, target = sample
        input.requires_grad = False

        # flush accumulators
        optimiser.zero_grad()

        # forward pass
        output = model.forward(input.to(device))

        # loss calculation
        loss = loss_function(
            output.to(device),
            torch.argmax(target, dim=1).to(device)
        )

        # backward pass
        loss.backward()

        # update
        optimiser.step()

        if experiment_helper.should_trigger(i): 
            train_target_list.append(target.to(device))
            train_output_list.append(output)

    # set model to evaluation mode
    model.eval()

    # Do a loss check on val set per epoch
    if experiment_helper.should_trigger(i):
        val_output_list = []
        val_target_list = []
        for batch_ndx, sample in enumerate(DataLoader(validation_dataset, batch_size=1)):
            with torch.no_grad():
                input, target = sample

                output = model.forward(input.to(device))

                val_output_list.append(output)
                val_target_list.append(target.to(device))

        val_output_list = torch.cat(val_output_list, dim=0)
        val_target_list = torch.cat(val_target_list, dim=0)
        train_output_list = torch.cat(train_output_list, dim=0)
        train_target_list = torch.cat(train_target_list, dim=0)

        # validate model
        experiment_helper.validate(
            loss_function,
            val_output_list,
            val_target_list,
            train_output_list,
            train_target_list,
            i
        )

        # save model weights
        if experiment_helper.is_progress():
            experiment_helper.save_checkpoint(model.state_dict)
        else:
            continue
# ============================================================================
