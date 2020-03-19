from transformers.transformer_factory import TransformerFactory
from dataset.dataset_factory import DatasetFactory
from optimisers.optimiser_factory import OptimiserFactory
from models.model_factory import ModelFactory
from losses.loss_factory import LossFactory
from torch.utils.data import DataLoader
from utils.config_parser import (get_config_data)
from progress.bar import ChargingBar
from utils.check_gpu import is_gpu_available

is_gpu_available()

config = get_config_data('./config/experiment_1.yml')

image_transformer = TransformerFactory(pipe_type="image")
optimiser_factory = OptimiserFactory()
dataset_factory = DatasetFactory(org_data_dir='./data')
model_factory = ModelFactory()
loss_factory = LossFactory()

# for i in range(5):
#     print( "fold: ", str(i) )
#     train_data = dataset_factory.get_dataset(
#         "train",
#         "fgvc7",
#         image_transformer,
#         i
#     )
#     print( len(train_data) )
#     val_data = dataset_factory.get_dataset(
#         "val",
#         "fgvc7",
#         image_transformer,
#         i
#     )
#     print( len(val_data) )

# test_data = dataset_factory.get_dataset(
#     "test",
#     "fgvc7"
# )
# print( len(test_data) )

# for batch_ndx, sample in enumerate(DataLoader(train_data)):
#     print(sample.size())

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
)

optimiser = optimiser_factory.get_optimiser(
    config['optimiser'], model.parameters(), config['hyper_params'])

loss_function = loss_factory.get_loss_function(
    config['loss_function']
)

# loop
bar = ChargingBar('Iteration', max=config["iteration"])
for i in range(config["iteration"]):
    bar.next()
    for batch_ndx, sample in enumerate(DataLoader(dataset, batch_size=1)):
        input = sample[0]
        target = sample[1]

        # forward pass
        output = model.forward(input)
        
        # loss calculation
        loss = loss_function(output,  target)
        
        # backward pass
        loss.backward()

        # update
        optimiser.step()
bar.finish()

# ============================================================================
