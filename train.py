from dataloader.dataloader_factory import CustomDataLoader

customDataLoader = CustomDataLoader()

traindata = customDataLoader.get_training_dataloader()

for batch_ndx, sample in enumerate(traindata):
    print(sample)