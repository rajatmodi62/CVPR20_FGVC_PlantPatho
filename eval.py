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
    model_factory = ModelFactory()
    evaluation_helper = EvaluationHelper(
        config['experiment_name'],
        True,
    )

    # ==================== Model testing / evaluation  setup ========================

    test_dataset = dataset_factory.get_dataset(
        'test',
        config['test_dataset']['name'],
        TransformerFactory(
            height=config['test_dataset']['resize_dims'],
            width=config['test_dataset']['resize_dims'],
        )
    )

    # ===============================================================================

    # ===================== Model testing / evaluation  loop ========================

    for model_name in config['model_list']:
        model = model_factory.get_model(
            model_name['model']['name'],
            model_name['model']['num_classes'],
            model_name['model']['pred_type'],
            model_name['model']['hyper_params'],
        ).to(device)
        weight_path = path.join(
            'results', model_name['model']['path'], 'weights.pth')
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
            test_dataset.get_csv_path(),
            test_output_list,
            model_name['model']['path']
        )

    if config['ensemble']:
        # use the helper to ensemble if needed
        pass

    # ===============================================================================
