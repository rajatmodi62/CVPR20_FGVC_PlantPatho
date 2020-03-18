from transformers.transformer_factory import TransformerFactory
from dataset.dataset_factory import DatasetFactory
from optimisers.optimiser_factory import OptimiserFactory
from torch.utils.data import DataLoader

image_transformer = TransformerFactory(pipe_type="image")
optimiser = OptimiserFactory()
dataset = DatasetFactory('./data')

for i in range(5):
    print( "fold: ", str(i) )
    train_data = dataset.get_dataset(
        "train",
        "fgvc7",
        image_transformer,
        i
    )
    print( len(train_data) )
    val_data = dataset.get_dataset(
        "val",
        "fgvc7",
        image_transformer,
        i
    )
    print( len(val_data) )

test_data = dataset.get_dataset(
    "test",
    "fgvc7"
)
print( len(test_data) )

# for batch_ndx, sample in enumerate(DataLoader(train_data)):
#     print(sample.size())

optim = optimiser.get_optimiser("RMSprop")
