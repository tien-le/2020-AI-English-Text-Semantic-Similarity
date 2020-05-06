# coding=utf-8
from __future__ import absolute_import, division, print_function
import os
import sys

sys.path.append(os.path.abspath('.'))
os.chdir(sys.path[0])

import argparse
import glob
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from src.transformers import (MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
                              WEIGHTS_NAME,
                              AutoConfig,
                              AutoModelForSequenceClassification,
                              AutoTokenizer,
                              HfArgumentParser,
                              TrainingArguments
                              )

from src.transformers import glue_convert_examples_to_features as convert_examples_to_features
from src.transformers import glue_output_modes as output_modes
from src.transformers import glue_processors as processors
from src.cores.albert_evaluate import *
from src.cores.albert_predict import *
from src.cores.albert_train import *

from src.libs.utils import Utils

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in MODEL_CONFIG_CLASSES), (), )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_and_cache_examples(args, task, tokenizer, evaluate=False, predict=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset,
        # and the others will use the cache
        torch.distributed.barrier()

    processor = processors[task]()
    output_mode = output_modes[task]

    key = 'train'
    if evaluate:
        key = 'dev'
    if predict:
        key = 'test'

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(key,
                                    list(filter(None, args.model_name_or_path.split("/"))).pop(),
                                    str(args.max_seq_length),
                                    str(task),
                                    ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]

        if evaluate:
            examples = processor.get_dev_examples(args.data_dir)
        elif predict:
            examples = processor.get_test_examples(args.data_dir)
        else:
            examples = processor.get_train_examples(args.data_dir)

        features = convert_examples_to_features(
            examples, tokenizer, max_length=args.max_seq_length, label_list=label_list, output_mode=output_mode,
        )
        # if args.local_rank in [-1, 0]:
        #     logger.info("Saving features into cached file %s", cached_features_file)
        #     torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset,
        # and the others will use the cache
        torch.distributed.barrier()

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    return dataset


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default='albert-large-v1',
        metadata={"help": "Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS)}
    )
    model_type: str = field(
        default='albert',
        metadata={"help": "Model type selected in the list: " + ", ".join(MODEL_TYPES)})
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pre-trained models downloaded from s3"}
    )


@dataclass
class DataProcessingArguments:
    task_name: str = field(
        default='sts-b',
        metadata={"help": "The name of the task to train selected in the list: " + ", ".join(processors.keys())}
    )
    data_dir: str = field(
        default='../data/input',
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer "
                          "than this will be truncated, sequences shorter will be padded."
                  },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def main(args):
    """

    :param args:
    :return:
    """
    for index in range(5):
        args.output_dir = '../../data/output_' + str(index)
        args.data_dir = '../../data/fold_' + str(index)

        if os.path.exists(args.output_dir) is False:
            os.makedirs(args.output_dir)
        if (os.path.exists(args.output_dir) and os.listdir(
                args.output_dir) and args.do_train and not args.overwrite_output_dir):
            shutil.rmtree(args.output_dir)
            raise ValueError(
                f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )

        # Setup CUDA, GPU & distributed training
        if args.local_rank == -1 or args.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
        else:
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            args.n_gpu = 1
        args.device = device

        # Setup logging
        logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                            datefmt="%m/%d/%Y %H:%M:%S",
                            level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
                            )
        logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                       args.local_rank,
                       device,
                       args.n_gpu,
                       bool(args.local_rank != -1),
                       args.fp16,
                       )
        # Set seed
        set_seed(args)

        # Prepare GLUE task
        args.task_name = args.task_name.lower()
        if args.task_name not in processors:
            raise ValueError("Task not found: %s" % args.task_name)
        processor = processors[args.task_name]()
        args.output_mode = output_modes[args.task_name]
        label_list = processor.get_labels()
        num_labels = len(label_list)

        # Load pretrained model and tokenizer
        if args.local_rank not in [-1, 0]:
            # Make sure only the first process in distributed training will download model & vocab
            torch.distributed.barrier()

        args.model_type = args.model_type.lower()
        config = AutoConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, cache_dir=args.cache_dir,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )

        if args.local_rank == 0:
            # Make sure only the first process in distributed training will download model & vocab
            torch.distributed.barrier()

        model.to(args.device)
        logger.info("Training/evaluation parameters %s", args)

        # Training
        if args.do_train:
            train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
            global_step, tr_loss = train(args, train_dataset, model, tokenizer)
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            # Create output directory if needed
            if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(args.output_dir)

            logger.info("Saving model checkpoint to %s", args.output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

            # Load a trained model and vocabulary that you have fine-tuned
            model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
            tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
            model.to(args.device)

        # Evaluation
        results = {}
        if args.do_eval and args.local_rank in [-1, 0]:
            tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c) for c in
                    sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
                logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            logger.info("Evaluate the following checkpoints: %s", checkpoints)
            for checkpoint in checkpoints:
                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

                model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
                model.to(args.device)
                result = evaluate(args, model, tokenizer, prefix=prefix)
                result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
                results.update(result)

        if args.do_pred and args.local_rank in [-1, 0]:
            tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c) for c in
                    sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
                logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            logger.info("Evaluate the following checkpoints: %s", checkpoints)
            for checkpoint in checkpoints:
                model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
                model.to(args.device)
                predict(args, model, tokenizer, prefix=str(index))


def run():
    # ModelArguments.model_type = 'albert'
    # ModelArguments.model_name_or_path = 'albert-large-v1'
    # ModelArguments.task_name = 'sts-b'
    # DataProcessingArguments.data_dir = '../data/input'
    # TrainingArguments.output_dir = '../data/output'

    parser = HfArgumentParser((ModelArguments, DataProcessingArguments, TrainingArguments))
    model_args, dataprocessing_args, training_args = parser.parse_args_into_dataclasses()

    # For now, let's merge all the sets of args into one,
    # but soon, we'll keep distinct sets of args, with a cleaner separation of concerns.
    args = argparse.Namespace(**vars(model_args), **vars(dataprocessing_args), **vars(training_args))

    args.do_train = True
    args.per_gpu_train_batch_size = 24
    args.do_eval = True
    args.per_gpu_eval_batch_size = 24
    args.do_pred = True
    args.per_gpu_pred_batch_size = 24

    args.model_type = 'albert'
    args.model_name_or_path = 'albert-xlarge-v1'
    args.max_seq_length = 128
    args.learning_rate = 2e-5
    args.num_train_epochs = 3
    args.from_tf = True
    args.overwrite_output_dir = '../../data/output'
    args.pseudo_labelling = True

    main(args=args)
    Utils().generate_keys_csv()
    if args.pseudo_labelling:
        for index in range(5):
            args.data_dir = '../../data/fold_' + str(index)
            keys_csv_data = pd.read_csv(os.path.join(args.data_dir, 'keys.csv'), names=['index', 'score'], sep=',')
            test_tsv_data = pd.read_csv(os.path.join(args.data_dir, 'test.tsv'), names=['text_a', 'text_b'], sep='\t')
            train_tsv_data = pd.read_csv(os.path.join(args.data_dir, 'train.tsv'), names=['text_a', 'text_b', 'score'],
                                         sep='\t')
            test_data = pd.concat([keys_csv_data, test_tsv_data], axis=1, ignore_index=True)
            test_data.reset_index(inplace=True, drop=True)
            test_data.columns = ['before_index', 'score', 'text_a', 'text_b']
            data = pd.concat([test_data[['text_a', 'text_b', 'score']], train_tsv_data], axis=0, ignore_index=True)

            data.to_csv(os.path.join(args.data_dir, 'train.tsv'), sep='\t', index=None, header=None)

    # args.num_train_epochs = 3
    # main(args=args)
    # Utils().generate_keys_csv()


if __name__ == "__main__":
    util = Utils()

    util.generate_fold_train_dev(train_csv_path='../../data/input/train.csv',
                                 test_csv_path='../../data/input/test.csv',
                                 fold_dir='../../data/fold')
    run()
