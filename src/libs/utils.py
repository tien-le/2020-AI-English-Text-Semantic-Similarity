# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
import sys
from rouge import Rouge

sys.path.append(os.path.abspath('.'))
os.chdir(sys.path[0])
import re
import copy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
import os
import torch
from sklearn.utils import shuffle
from datetime import datetime
from pytorch_lightning.logging import TestTubeLogger


class Utils(object):
    def __init__(self):
        pass

    def match(self, data):
        result = pd.DataFrame(columns=['label'], data=data['label'])

        text_as = data['text_a'].values.tolist()
        text_bs = data['text_b'].values.tolist()

        rouge = Rouge()
        for a_index in range(len(text_as)):
            rouge_score_std = rouge.get_scores(text_as[a_index], text_bs[a_index])
            rouge_score_std = rouge_score_std[0]['rouge-l']['f']
            for b_index in range(len(text_bs)):
                if a_index != b_index:
                    rouge_score_cur = rouge.get_scores(text_as[a_index], text_bs[b_index])
                    rouge_score_cur = rouge_score_cur[0]['rouge-l']['f']
                    if rouge_score_cur > rouge_score_std:
                        tmp = text_bs[b_index]
                        text_bs[b_index] = text_bs[a_index]
                        text_bs[a_index] = tmp
                        rouge_score_std = rouge_score_cur

        result['text_a'] = text_as
        result['text_b'] = text_bs

        return result

    def generate_train_dev_test(self, train_csv_path, test_csv_path, max_sequence_length):
        """
        生成训练集和测试集
        :param train_csv_path:
        :param test_csv_path:
        :return:
        """
        train_csv_data = pd.read_csv(train_csv_path, encoding='utf-8')
        test_csv_data = pd.read_csv(test_csv_path, encoding='utf-8')

        train_tsv_data = [['', '', '']]
        train_csv_last_data = [[]]
        for index in range(train_csv_data.shape[0]):
            assert train_csv_data.shape[1] == 3
            text_a = train_csv_data.iloc[index, 0]
            text_b = train_csv_data.iloc[index, 1]
            text_a_length = len(text_a)
            text_b_length = len(text_b)
            label = train_csv_data.iloc[index, 2]

            if (text_a_length + text_b_length) < (max_sequence_length - 3):
                train_csv_last_data.append([text_a, text_b, label])
                if label == '':
                    print('error')
                    print(text_a)
                    print(text_b)
            else:
                text_a_split = text_a.split(',')
                text_b_split = text_b.split(',')
                text_a_split_length = len(text_a_split)
                text_b_split_length = len(text_b_split)
                if text_a_split_length != text_b_split_length:
                    if label == '':
                        print('error')
                        print(text_a)
                        print(text_b)

                    train_csv_last_data.append([text_a, text_b, label])
                    continue

                label_split = [label] * text_a_split_length

                for current_index in range(text_a_split_length):
                    phrase_text_a = text_a_split[current_index]
                    phrase_text_b = text_b_split[current_index]
                    phrase_text_a_length = len(phrase_text_a)
                    phrase_text_b_length = len(phrase_text_b)
                    phrase_label = label_split[current_index]

                    before_length = len(''.join(train_tsv_data[-1]))
                    if (before_length + phrase_text_a_length + phrase_text_b_length) < (max_sequence_length - 3):
                        if before_length == 0:
                            train_tsv_data[-1] = [phrase_text_a, phrase_text_b, str(phrase_label)]
                        else:
                            train_tsv_data[-1][0] += phrase_text_a
                            train_tsv_data[-1][1] += phrase_text_b
                    else:
                        train_tsv_data.append([phrase_text_a, phrase_text_b, str(phrase_label)])

        while [] in train_csv_last_data:
            train_csv_last_data.remove([])

        while [] in train_tsv_data:
            train_tsv_data.remove([])

        train_tsv_data.extend(train_csv_last_data)
        train_tsv_data = np.asarray(train_tsv_data)
        result = pd.DataFrame(data=train_tsv_data[:, 0], columns=['text_a'])
        result['text_b'] = train_tsv_data[:, 1]
        result['label'] = np.asarray(train_tsv_data[:, 2]).astype(float)

        train = pd.DataFrame()
        dev = pd.DataFrame()
        for name, group in result.groupby('label'):
            group = self.match(group)
            if group.shape[0] < 5:
                train = pd.concat([train, group], ignore_index=True)
                continue

            current_train, current_dev = train_test_split(group, test_size=0.2, random_state=42, shuffle=True)

            train = pd.concat([train, current_train], ignore_index=True)
            dev = pd.concat([dev, current_dev], ignore_index=True)

        from sklearn.utils import shuffle
        train = shuffle(train)
        dev = shuffle(dev)

        train[['text_a', 'text_b', 'label']].to_csv(path_or_buf='../../data/fold/train.tsv', sep='\t', header=None,
                                                    index=None)
        dev[['text_a', 'text_b', 'label']].to_csv(path_or_buf='../../data/fold/dev.tsv', sep='\t', header=None,
                                                  index=None)

    @staticmethod
    def replace_typical_misspell(text):
        mispell_dict = {'colour': 'color', 'centre': 'center', 'didnt': 'did not', 'doesnt': 'does not',
                        'isnt': 'is not', 'shouldnt': 'should not', 'favourite': 'favorite', 'travelling': 'traveling',
                        'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',
                        'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize',
                        'instagram': 'social medium',
                        'whatsapp': 'social medium', 'snapchat': 'social medium', "ain't": "is not",
                        "aren't": "are not", "can't": "cannot", "'cause": "because", "could've": "could have",
                        "couldn't": "could not", "didn't": "did not", "doesn't": "does not", "don't": "do not",
                        "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would",
                        "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you",
                        "how'll": "how will", "how's": "how is", "I'd": "I would", "I'd've": "I would have",
                        "I'll": "I will", "I'll've": "I will have", "I'm": "I am", "I've": "I have",
                        "i'd": "i would", "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have",
                        "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                        "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
                        "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not",
                        "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
                        "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
                        "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock",
                        "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                        "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
                        "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
                        "she's": "she is", "should've": "should have", "shouldn't": "should not",
                        "shouldn't've": "should not have",
                        "so've": "so have", "so's": "so as", "this's": "this is", "that'd": "that would",
                        "that'd've": "that would have", "that's": "that is",
                        "there'd": "there would", "there'd've": "there would have", "there's": "there is",
                        "here's": "here is", "they'd": "they would", "they'd've": "they would have",
                        "they'll": "they will", "they'll've": "they will have",
                        "they're": "they are", "they've": "they have", "to've": "to have",
                        "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
                        "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                        "we've": "we have", "weren't": "were not", "what'll": "what will",
                        "what'll've": "what will have", "what're": "what are", "what's": "what is",
                        "what've": "what have", "when's": "when is", "when've": "when have",
                        "where'd": "where did", "where's": "where is", "where've": "where have",
                        "who'll": "who will", "who'll've": "who will have", "who's": "who is",
                        "who've": "who have", "why's": "why is", "why've": "why have",
                        "will've": "will have", "won't": "will not", "won't've": "will not have",
                        "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                        "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have",
                        "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
                        "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                        "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center',
                        'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling',
                        'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',
                        'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize',
                        'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What',
                        'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are',
                        'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do',
                        'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does',
                        'mastrubation': 'masturbation', 'mastrubate': 'masturbate',
                        "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum',
                        'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018',
                        'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess',
                        "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization',
                        'demonitization': 'demonetization', 'demonetisation': 'demonetization'
                        }

        def _replace(match):
            return mispellings[match.group(0)]

        def _get_mispell(mispell_dict):
            mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
            return mispell_dict, mispell_re

        mispellings, mispellings_re = _get_mispell(mispell_dict)

        return mispellings_re.sub(_replace, text.lower())

    @staticmethod
    def generate_fold_train_dev(fold_dir, train_csv_path, test_csv_path):
        data_result_list = [[], [], [], [], []]

        data = pd.read_csv(filepath_or_buffer=train_csv_path, encoding='utf-8')
        data = pd.concat([data[['text_a', 'text_b', 'socre']], data[['text_b', 'text_a', 'socre']]], ignore_index=True)
        data.sort_values(by=['socre'], inplace=True)
        data.reset_index(drop=True, inplace=True)
        data = data[['text_a', 'text_b', 'socre']]

        for index in range(0, data.shape[0], 5):
            data_result_list[0].append([data.iloc[index, 0], data.iloc[index, 1], data.iloc[index, 2]])
            if index + 1 >= data.shape[0]:
                break
            data_result_list[1].append([data.iloc[index + 1, 0], data.iloc[index + 1, 1], data.iloc[index + 1, 2]])
            data_result_list[2].append([data.iloc[index + 2, 0], data.iloc[index + 2, 1], data.iloc[index + 2, 2]])
            data_result_list[3].append([data.iloc[index + 3, 0], data.iloc[index + 3, 1], data.iloc[index + 3, 2]])
            data_result_list[4].append([data.iloc[index + 4, 0], data.iloc[index + 4, 1], data.iloc[index + 4, 2]])

        for index in range(5):
            train_data = copy.deepcopy(data_result_list)
            dev_data = train_data[index]
            dev_data = np.asarray(dev_data)

            dev_tsv_data = pd.DataFrame(data=dev_data[:, 0].tolist(), columns=['text_a'])
            dev_tsv_data['text_b'] = dev_data[:, 1]
            dev_tsv_data['label'] = dev_data[:, 2]

            train_data.remove(train_data[index])
            train_data = np.concatenate(train_data)
            train_data = np.asarray(train_data)
            train_tsv_data = pd.DataFrame(data=train_data[:, 0].tolist(), columns=['text_a'])
            train_tsv_data['text_b'] = train_data[:, 1]
            train_tsv_data['label'] = train_data[:, 2]

            test_tsv_data = pd.read_csv(test_csv_path)
            # 改成小写
            train_tsv_data['text_a'] = train_tsv_data['text_a'].apply(lambda a: Utils.replace_typical_misspell(a))
            train_tsv_data['text_b'] = train_tsv_data['text_b'].apply(lambda a: Utils.replace_typical_misspell(a))
            dev_tsv_data['text_a'] = dev_tsv_data['text_a'].apply(lambda a: Utils.replace_typical_misspell(a))
            dev_tsv_data['text_b'] = dev_tsv_data['text_b'].apply(lambda a: Utils.replace_typical_misspell(a))
            test_tsv_data['text_a'] = test_tsv_data['text_a'].apply(lambda a: Utils.replace_typical_misspell(a))
            test_tsv_data['text_b'] = test_tsv_data['text_b'].apply(lambda a: Utils.replace_typical_misspell(a))

            fold_index_dir = os.path.join(fold_dir + '_' + str(index))
            if os.path.exists(fold_index_dir) is False:
                os.makedirs(fold_index_dir)

            dev_tsv_data = shuffle(dev_tsv_data)
            train_tsv_data = shuffle(train_tsv_data)

            test_tsv_data.to_csv(os.path.join(fold_index_dir, 'test.tsv'), header=None, index=None, sep='\t',
                                 encoding='utf-8')
            dev_tsv_data[['text_a', 'text_b', 'label']].to_csv(os.path.join(fold_index_dir, 'dev.tsv'), header=None,
                                                               index=None, sep='\t',
                                                               encoding='utf-8')
            train_tsv_data[['text_a', 'text_b', 'label']].to_csv(os.path.join(fold_index_dir, 'train.tsv'), header=None,
                                                                 index=None, sep='\t',
                                                                 encoding='utf-8')

    @staticmethod
    def generate_keys_csv():
        result = None
        for index in range(5):
            current_data = pd.read_csv('../../data/fold_' + str(index) + '/keys.csv', header=None)
            current_data.columns = ['a', 'b']
            if result is None:
                result = current_data
            else:
                result += current_data

        result /= 5
        result['a'] = result['a'].astype(int)
        result.to_csv('../../data/fold/key.csv', index=None, header=None)


class LongformerUtils(object):
    @staticmethod
    def setup_testube_logger() -> TestTubeLogger:
        """ Function that sets the TestTubeLogger to be used. """
        try:
            job_id = os.environ["SLURM_JOB_ID"]
        except Exception:
            job_id = None

        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y--%H-%M-%S")

        return TestTubeLogger(
            save_dir="../../data/experiments/",
            version=job_id + "_" + dt_string if job_id else dt_string,
            name="lightning_logs",
        )

    @staticmethod
    def mask_fill(fill_value: float, tokens: torch.tensor, embeddings: torch.tensor,
                  padding_index: int, ) -> torch.tensor:
        """
        Function that masks embeddings representing padded elements.
        :param fill_value: the value to fill the embeddings belonging to padded tokens.
        :param tokens: The input sequences [bsz x seq_len].
        :param embeddings: word embeddings [bsz x seq_len x hiddens].
        :param padding_index: Index of the padding token.
        """
        padding_mask = tokens.eq(padding_index).unsqueeze(-1)

        return embeddings.float().masked_fill_(padding_mask, fill_value).type_as(embeddings)

    @staticmethod
    def generate_train_dev_csv(input_train_path, output_train_path, output_dev_path):
        """
        生成训练/验证/测试/标签 json文件
        :return:
        """
        if os.path.exists(output_train_path) and os.path.exists(output_dev_path):
            train = pd.read_csv(output_train_path, encoding='utf-8')
            dev = pd.read_csv(output_dev_path, encoding='utf-8')
        else:
            # text, label
            data = pd.read_csv(filepath_or_buffer=input_train_path, encoding='utf-8')

            data['text'] = data['text'].apply(lambda x: x.replace('\n', ' '))
            data['text'] = data['text'].apply(lambda x: x.replace('\t', ' '))
            data.drop_duplicates(inplace=True)
            data.reset_index(inplace=True, drop=True)

            data_count = data.groupby(by=['label'], as_index=False)['label'].agg({'number': 'count'})
            data_count = dict(zip(data_count['label'], data_count['number']))
            data_count = dict(sorted(data_count.items(), key=lambda item: item[1]))
            count_average = int(data.shape[0] // (len(data_count)))
            # 1646
            print('count_average: {}'.format(count_average))

            # 不用处理的label
            no_deal_label = []
            deal_label = []
            deal_label_number = []
            for key, value in data_count.items():
                if value > count_average:
                    no_deal_label.append(key)
                else:
                    deal_label.append(key)
                    deal_label_number.append(count_average // data_count[key])

            labels = []
            texts = []
            for index in range(data.shape[0]):
                text = data.iloc[index, 0]
                label = data.iloc[index, 1]

                # 不截断
                texts.append(text)
                labels.append(label)

                if len(text) < 5:
                    continue

            data = pd.DataFrame({'text': texts, 'label': labels})

            train_index = []
            dev_index = []
            for item in data['label'].unique():
                item_data = data[data['label'] == item]
                item_data = item_data.index.tolist()
                item_data_len = len(item_data)
                train_count = int(item_data_len * 0.8)
                train_count_data = random.sample(item_data, train_count)
                train_index.extend(train_count_data)
                for value in train_count_data:
                    item_data.remove(value)
                dev_index.extend(item_data)

            train = data.iloc[np.asarray(train_index)]
            dev = data.iloc[np.asarray(dev_index)]

            train = shuffle(train, random_state=42)
            dev = shuffle(dev, random_state=42)

            train.reset_index(inplace=True, drop=True)
            dev.reset_index(inplace=True, drop=True)

            train.to_csv(output_train_path, encoding='utf-8', index=None)
            dev.to_csv(output_dev_path, encoding='utf-8', index=None)

        return train, dev

# if __name__ == '__main__':
#     util = Utils()
#     #
#     #     # util.generate_train_dev_test(
#     #     #     train_csv_path='../../data/input/train.csv',
#     #     #     test_csv_path='../../data/input/test.csv',
#     #     #     max_sequence_length=128
#     #     # )
#     #
#     util.generate_fold_train_dev(train_csv_path='../../data/input/train.csv', test_csv_path='../../data/input/test.csv',
#                                  fold_dir='../../data/fold')
# #
# #     # util.generate_keys_csv()
