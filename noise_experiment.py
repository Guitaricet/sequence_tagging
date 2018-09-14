import os
import pickle
import logging

import numpy as np
import pandas as pd

from model.data_utils import CoNLLDataset, get_char_vocab
from model.ner_model import NERModel
from model.config import Config


formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M'
)
logger = logging.getLogger()
logger.setLevel('DEBUG')

N_TRAINS = 10
NOISE_LEVELS = np.concatenate([np.arange(0, 0.05, 0.005), np.arange(0.05, 0.2, 0.01)])


if __name__ == '__main__':
    from time import time
    import argparse

    time_total = time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--rove-path', type=str)
    parser.add_argument('--results-filename', type=str, default='noise_experiment_results.csv')

    args = parser.parse_args()
    results_save_path = args.results_filename
    rove_path = args.rove_path

    if os.path.exists(results_save_path):
        logging.warning('File at path %s exists' % results_save_path)
        yes = input('Replace it? (y/n) ')
        if yes.lower() == 'y':
            logging.info('File %s will be replaced' % results_save_path)
        else:
            logging.info('Canceling execution')
            exit(1)

    dataset = args.dataset.lower()
    if dataset == 'conll':
        basepath = '../ner/data/conll2003/splits/'
        train_path = basepath + 'train.txt'
        val_path = basepath + 'valid.txt'
        test_path = basepath + 'test.txt'
    elif dataset == 'collection5':
        basepath = '../ner/data/collection5/splits/'
        train_path = basepath + 'train.txt'
        val_path = basepath + 'valid.txt'
        test_path = basepath + 'test.txt'
    elif dataset == 'CAp':
        raise NotImplementedError
    else:
        raise ValueError(args.dataset)

    split_by = '\t'
    with open(os.path.join(rove_path, 'chars_vocab.pkl'), 'rb') as f:
        _, vocab_chars = pickle.load(f)

    config = Config()
    config.use_chars = False
    proc_tag = config.processing_tag

    train_dataset = CoNLLDataset(train_path, char_vocab=vocab_chars, split_by=split_by, processing_tag=proc_tag)
    val_dataset = CoNLLDataset(val_path, char_vocab=vocab_chars, split_by=split_by, processing_tag=proc_tag)
    test_dataset = CoNLLDataset(test_path, char_vocab=vocab_chars, split_by=split_by, processing_tag=proc_tag)

    logging.info(f'first element of train dataset: {iter(train_dataset)}')

    results = []

    for _ in range(N_TRAINS):
        for noise_level in NOISE_LEVELS:
            logging.info(f'training for noise level {noise_level}')
            train_dataset.noise_level = noise_level
            val_dataset.noise_level = noise_level
            test_dataset.noise_level = noise_level

            model = NERModel(config,
                             is_rove=True,
                             rove_path=rove_path,
                             bme_encoding_size=7*len(vocab_chars))  # 7*len(vocab) is rove structure
            model.build()

            model.train(train_dataset, val_dataset)
            metrics = model.evaluate(test_dataset)
            logger.info(f"F1 on noised testset: {metrics['f1']}")
            results.append({'f1_test': metrics['f1'],
                            'noise_level_test': noise_level,
                            'noise_level_train': noise_level})
            pd.DataFrame(results).to_csv(results_save_path)

            logger.info(f"F1 on original testset: {metrics['f1']}")
            test_dataset.noise_level = 0
            metrics_orig = model.evaluate(test_dataset)
            results.append({'f1_test': metrics_orig['f1'],
                            'noise_level_test': -1,
                            'noise_level_train': noise_level})
            pd.DataFrame(results).to_csv(results_save_path)
