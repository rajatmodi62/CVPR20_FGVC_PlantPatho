import torch
from os import path
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
from transformers.transformer_factory import TransformerFactory
from dataset.dataset_factory import DatasetFactory
from optimisers.optimiser_factory import OptimiserFactory
from models.model_factory import ModelFactory
from losses.loss_factory import LossFactory
from torch.utils.data import DataLoader
from utils.config_parser import (get_config_data)
from utils.experiment_utils import (model_checkpoint, experiment_dir_creator)
from utils.check_gpu import get_training_device

parser = argparse.ArgumentParser()
parser.add_argument("experiment_file",
                    help="The name of the experiment config file")
args = parser.parse_args()

# Get experiment config values
if args.experiment_file is None:
    exit()
config = get_config_data(path.join('config', args.experiment_file))

# Get GPU / CPU device instance
device = get_training_device()

# Create pipeline objects
optimiser_factory = OptimiserFactory()
dataset_factory = DatasetFactory(org_data_dir='./data')
model_factory = ModelFactory()
loss_factory = LossFactory()

# ===================== Model training / validation ==========================

# setup
experiment_dir_creator(config['experiment_name'], rewrite=True)

training_dataset = dataset_factory.get_dataset(
    'train',
    config['dataset']['name'],
    TransformerFactory(
        height=config['resize_dims'],
        width=config['resize_dims'],
        pipe_type="image",
    ),
    config['dataset']['fold']
)

validation_dataset = dataset_factory.get_dataset(
    'val',
    config['dataset']['name'],
    TransformerFactory(
        height=config['resize_dims'],
        width=config['resize_dims'],
    ),
    config['dataset']['fold']
)

model = model_factory.get_model(
    config['model']['name']
).to(device)

optimiser = optimiser_factory.get_optimiser(
    config['optimiser'], model.parameters(), config['hyper_params'])

loss_function = loss_factory.get_loss_function(
    config['loss_function']
)

# loop
batch_size = config["batch_size"]

for i in trange(config["epochs"], desc="Epochs : "):
    # set model to training mode
    model.train()

    train_loss = 0
    for batch_ndx, sample in enumerate(DataLoader(training_dataset, batch_size=batch_size)):
        input, target = sample
        input.requires_grad = False
        target = torch.argmax(target, dim=1)

        # flush accumulators
        optimiser.zero_grad()

        # forward pass
        output = model.forward(input.to(device))

        # loss calculation
        train_loss = loss_function(
            output.to('cpu'),
            target.to('cpu')
        )

        # backward pass
        train_loss.backward()

        # update
        optimiser.step()

    # set model to evaluation mode
    model.eval()

    # Do a loss check on val set per epoch
    val_acc = 0.
    val_loss = 0.
    for batch_ndx, sample in enumerate(DataLoader(validation_dataset, batch_size=1)):
        with torch.no_grad():
            input, target = sample
            target = torch.argmax(target, dim=1)

            output = model.forward(input.to(device))
            loss = loss_function(
                output.to('cpu'),
                target.to('cpu')
            )

            val_acc += (target.item() == torch.argmax(output, dim=1).item())
            print(loss)
            val_loss += loss.item()

    val_loss /= len(validation_dataset)
    val_acc /= len(validation_dataset)

    print(val_loss)

    # save model params
    model_checkpoint(
        config["experiment_name"],
        val_loss,
        train_loss.item(),
        val_acc,
        i,
        model.state_dict()
    )
# ============================================================================
