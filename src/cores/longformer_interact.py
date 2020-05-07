"""
Runs a script to interact with a model using the shell.
"""
import os
import torch
import pandas as pd

from test_tube import HyperOptArgumentParser
from src.cores.longformer_classifier import LONGFORMERClassifier

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_model_from_experiment(experiment_folder: str):
    """ Function that loads the model from an experiment folder.
    :param experiment_folder: Path to the experiment folder.
    Return:
        - Pretrained model.
    """
    tags_csv_file = experiment_folder + "/meta_tags.csv"
    tags = pd.read_csv(tags_csv_file, header=None, index_col=0, squeeze=True).to_dict()
    checkpoints = [
        file
        for file in os.listdir(experiment_folder + "/checkpoints/")
        if file.endswith(".ckpt")
    ]
    checkpoint_path = experiment_folder + "/checkpoints/" + checkpoints[-1]

    model = LONGFORMERClassifier.load_from_metrics(weights_path=checkpoint_path, tags_csv=tags_csv_file)

    # Make sure model is in prediction mode
    model.eval()
    model.freeze()

    return model


if __name__ == "__main__":
    parser = HyperOptArgumentParser(description="Minimalist BERT Classifier", add_help=True)
    parser.add_argument("--experiment", default='../../data/experiments/lightning_logs/version_07-05-2020--11-28-30',
                        type=str, help="Path to the experiment folder.")
    hparams = parser.parse_args()
    print("Loading model...")
    model = load_model_from_experiment(hparams.experiment).to(device=DEVICE)

    print("Please write a movie review or quit to exit the interactive shell:")
    # Get input sentence

    test_path = '../../data/fold_0/test.tsv'
    test = pd.read_csv(test_path, encoding='utf-8', sep='\t', header=None)

    labels = []
    for index in range(0, test.shape[0], 4):
        if index % 1000 == 0:
            print(index)

        label = model.predict(
            samples=[{'text': test.iloc[index + 0, 0] + ' ' + test.iloc[index + 0, 1]},
                     {'text': test.iloc[index + 1, 0] + ' ' + test.iloc[index + 1, 1]},
                     {'text': test.iloc[index + 2, 0] + ' ' + test.iloc[index + 2, 1]},
                     {'text': test.iloc[index + 3, 0] + ' ' + test.iloc[index + 3, 1]}])['predicted_label']

        labels.extend(label)

    test['label'] = labels
    test['label'].to_csv('../../data/fold_0/keys.csv', header=None, encoding='utf-8')
