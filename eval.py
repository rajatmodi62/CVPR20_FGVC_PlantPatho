import torch
from torch.utils.data import DataLoader
from os import (path, environ)
from tqdm import (trange, tqdm)
from dataset.dataset_factory import DatasetFactory
from transformers.transformer_factory import TransformerFactory
from models.model_factory import ModelFactory
from utils.evaluation_utils import EvaluationHelper


def eval(config, device):
    # Create pipeline objects
    dataset_factory = DatasetFactory(org_data_dir='./data')

    transformer_factory = TransformerFactory()

    model_factory = ModelFactory()

    evaluation_helper = EvaluationHelper(
        config['experiment_name'],
        overwrite=True,
        ensemble=config['ensemble']
    )

    # ==================== Model testing / evaluation setup ========================

    test_dataset = dataset_factory.get_dataset(
        'test',
        config['test_dataset']['name'],
        transformer_factory.get_transformer(
            height=config['test_dataset']['resize_dims'],
            width=config['test_dataset']['resize_dims'],
        )
    )

    # ===============================================================================

    # ===================== Model testing / evaluation  loop ========================

    for experiment_item in config['experiment_list']:
        print("[ Experiment : ", experiment_item['experiment']['path'], " ]")

        model = model_factory.get_model(
            experiment_item['experiment']['name'],
            config['num_classes'],
            experiment_item['experiment']['pred_type'],
            experiment_item['experiment']['hyper_params'],
            tuning_type=None
        ).to(device)

        weight_path = path.join(
            'results', experiment_item['experiment']['path'], 'weights.pth')

        model.load_state_dict(torch.load(weight_path))

        model.eval()

        test_output_list = []
        for batch_ndx, sample in enumerate(tqdm(DataLoader(test_dataset, batch_size=1), desc="Samples : ")):
            with torch.no_grad():
                input = sample
                input = input.to(device)

                output = model.forward(input)

                test_output_list.append(output)

        test_output_list = torch.cat(test_output_list, dim=0)

        # use this list to write using a helper
        evaluation_helper.evaluate(
            experiment_item['experiment']['pred_type'],
            config['num_classes'],
            experiment_item['experiment']['path'],
            test_dataset.get_csv_path(),
            test_output_list
        )

    if config['ensemble']:
        evaluation_helper.ensemble(
            test_dataset.get_csv_path()
        )

    # ===============================================================================
