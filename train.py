from transformers.transformer_factory import Transformer
from dataset.dataset_factory import Dataset
from torch.utils.data import DataLoader

image_transformer = Transformer(pipe_type="image")
dataset = Dataset('data')

train_data = dataset.get_dataset(
    "train",
    "fgvc7",
    image_transformer,
    1
)

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
