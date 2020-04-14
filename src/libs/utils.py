# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
import sys
from rouge import Rouge

sys.path.append(os.path.abspath('.'))
os.chdir(sys.path[0])
import copy
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


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

    def generate_fold_train_dev(self, fold_dir, train_csv_path, test_csv_path):
        data_result_list = [[], [], [], [], []]

        data = pd.read_csv(filepath_or_buffer=train_csv_path, encoding='utf-8')
        data.sort_values(by=['socre'], inplace=True)
        for index in range(0, data.shape[0], 5):
            data_result_list[0].append([data.iloc[index, 0], data.iloc[index, 1], data.iloc[index, 2]])
            data_result_list[1].append([data.iloc[index + 1, 0], data.iloc[index + 1, 1], data.iloc[index + 1, 2]])
            data_result_list[2].append([data.iloc[index + 2, 0], data.iloc[index + 2, 1], data.iloc[index + 2, 2]])
            if index + 3 >= data.shape[0]:
                break
            data_result_list[3].append([data.iloc[index + 3, 0], data.iloc[index + 3, 1], data.iloc[index + 3, 2]])
            data_result_list[4].append([data.iloc[index + 4, 0], data.iloc[index + 4, 1], data.iloc[index + 4, 2]])

        for index in range(5):
            train_data = copy.deepcopy(data_result_list)
            dev_data = train_data[index]
            train_data.remove(train_data[index])

            dev_data = np.asarray(dev_data)

            dev_tsv_data = pd.DataFrame(data=dev_data[:, 0].tolist(), columns=['text_a'])
            dev_tsv_data['text_b'] = dev_data[:, 1]
            dev_tsv_data['label'] = dev_data[:, 2]

            train_data = np.concatenate((train_data[0], train_data[1], train_data[2], train_data[3]))

            train_data = np.asarray(train_data)
            train_tsv_data = pd.DataFrame(data=train_data[:, 0].tolist(), columns=['text_a'])
            train_tsv_data['text_b'] = train_data[:, 1]
            train_tsv_data['label'] = train_data[:, 2]

            fold_index_dir = os.path.join(fold_dir + '_' + str(index))
            test_tsv_data = pd.read_csv(test_csv_path)
            if os.path.exists(fold_index_dir) is False:
                os.makedirs(fold_index_dir)

            dev_tsv_data = shuffle(dev_tsv_data)
            train_tsv_data = shuffle(train_tsv_data)

            test_tsv_data.to_csv(os.path.join(fold_index_dir, 'test.tsv'), header=None, index=None, sep='\t',
                                 encoding='utf-8')
            dev_tsv_data.to_csv(os.path.join(fold_index_dir, 'dev.tsv'), header=None, index=None, sep='\t',
                                encoding='utf-8')
            train_tsv_data.to_csv(os.path.join(fold_index_dir, 'train.tsv'), header=None, index=None, sep='\t',
                                  encoding='utf-8')

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

    def generate_keys_csv(self):
        result = None
        for index in range(5):
            current_data = pd.read_csv('../../data/fold_' + str(index) + '/keys' + str(index) + '.csv', header=None)
            current_data.columns = ['a', 'b']
            if result is None:
                result = current_data
            else:
                result += current_data

        result /= 5
        result['a'] = result['a'].astype(int)
        result.to_csv('../../data/fold/key.csv', index=None, header=None)


if __name__ == '__main__':
    util = Utils()

    # util.generate_train_dev_test(
    #     train_csv_path='../../data/input/train.csv',
    #     test_csv_path='../../data/input/test.csv',
    #     max_sequence_length=128
    # )

    # util.generate_fold_train_dev(train_csv_path='../../data/input/train.csv', test_csv_path='../../data/input/test.csv',
    #                              fold_dir='../../data/fold')

    util.generate_keys_csv()
