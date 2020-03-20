import torch
from transformers.transformer_factory import TransformerFactory
from dataset.dataset_factory import DatasetFactory
from optimisers.optimiser_factory import OptimiserFactory
from models.model_factory import ModelFactory
from losses.loss_factory import LossFactory
from torch.utils.data import DataLoader
from utils.config_parser import (get_config_data)
from progress.bar import ChargingBar
from utils.check_gpu import get_training_device

device = get_training_device()

config = get_config_data('./config/experiment_2.yml')

image_transformer = TransformerFactory(
    pipe_type="image",
    height=100,
    width=100
)
optimiser_factory = OptimiserFactory()
dataset_factory = DatasetFactory(org_data_dir='./data')
model_factory = ModelFactory()
loss_factory = LossFactory()

# ========================== Model training ==================================

# setup
dataset = dataset_factory.get_dataset(
    'train',
    config['dataset']['name'],
    image_transformer,
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
batch_size = 16
bar = ChargingBar('Iteration', max= ( config["iteration"] * len(dataset) / batch_size ))
for i in range(config["iteration"]):
    for batch_ndx, sample in enumerate(DataLoader(dataset, batch_size=batch_size)):
        input, target = sample
        input.requires_grad = False

        # forward pass
        output = model.forward(input.to(device))

        # loss calculation
        # loss = loss_function(output,  target)

        # backward pass
        # loss.backward()

        # update
        # optimiser.step()
        bar.next()
bar.finish()

# ============================================================================
