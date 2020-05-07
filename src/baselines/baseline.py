import warnings

warnings.simplefilter('ignore')

import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import KFold

from simpletransformers.classification import ClassificationModel

train = pd.read_csv('../../data/input/train.csv')
test = pd.read_csv('../../data/input/test.csv')

train.columns = ['text_a', 'text_b', 'labels']
train_copy = train.copy(deep=True)
train_copy.columns = ['text_b', 'text_a', 'labels']
train = pd.concat([train, train_copy], axis=0, ignore_index=True)
train['text_a'] = train['text_a'].apply(lambda a: a.lower())
train['text_b'] = train['text_b'].apply(lambda a: a.lower())

train_args = {
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'num_train_epochs': 10,
    'regression': True,
    'fp16': False,
    'max_seq_length': 256
}

predictions = np.zeros(shape=(len(test),))

kfold = KFold(n_splits=5, shuffle=True, random_state=2020)
for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(train)):
    train_fold = train.iloc[trn_idx]
    valid_fold = train.iloc[val_idx]

    model = ClassificationModel('roberta',
                                model_name='../../data/roberta-base-pytorch',
                                num_labels=1,
                                use_cuda=True,
                                cuda_device=0,
                                args=train_args)

    model.train_model(train_fold, eval_df=valid_fold)

    text_list = list()
    for i, row in tqdm(test.iterrows()):
        text_list.append([row['text_a'], row['text_b']])

    pred, _ = model.predict(text_list)
    predictions += pred / kfold.n_splits

sub = pd.DataFrame()
sub['ID'] = test.index
sub['score'] = predictions
sub.head()

sub.score.describe()

sub.to_csv('roberta_10_epochs_5_folds_orig.csv', index=False)

sub.loc[sub.score < 0.08, 'score'] = 0
sub.loc[sub.score > 5, 'score'] = 5

sub.score.describe()

sub.to_csv('roberta_10_epochs_5_folds.csv', index=False, header=False)
