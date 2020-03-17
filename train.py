from transformers.transformer_factory import Transformer
from dataloader.dataloader_factory import CustomDataLoader

image_transformer = Transformer()
customDataLoader = CustomDataLoader('data')

train_dataloader_list, val_dataloader_list, test_dataloader = customDataLoader.get_dataset(
    "fgvc7", image_transformer)

for batch_ndx, sample in enumerate(test_dataloader):
    print(sample)
