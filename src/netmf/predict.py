#!/usr/bin/env python
# encoding: utf-8
# File Name: predict.py
# Author: Jiezhong Qiu
# Create Time: 2017/07/17 21:57
# TODO:

import argparse
import logging
import os
import pickle as pkl
import warnings
from typing import Optional

import numpy as np
import scipy.io
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import ShuffleSplit
from sklearn.multiclass import OneVsRestClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

logger = logging.getLogger(__name__)


def construct_indicator(y_score, y):
    # rank the labels by the scores directly
    num_label = np.sum(y, axis=1, dtype=np.int)
    y_sort = np.fliplr(np.argsort(y_score, axis=1))
    y_pred = np.zeros_like(y, dtype=np.int)
    for i in range(y.shape[0]):
        for j in range(num_label[i]):
            y_pred[i, y_sort[i, j]] = 1
    return y_pred


def load_w2v_feature(file):
    with open(file, "rb") as f:
        nu = 0
        for line in f:
            content = line.strip().split()
            nu += 1
            if nu == 1:
                n, d = int(content[0]), int(content[1])
                feature = [[] for i in range(n)]
                continue
            index = int(content[0])
            for x in content[1:]:
                feature[index].append(float(x))
    for item in feature:
        assert len(item) == d
    return np.array(feature, dtype=np.float32)


def load_label(matfile, variable_name="group"):
    data = scipy.io.loadmat(matfile)
    logger.info("loading mat file %s", matfile)
    label = data[variable_name].todense().astype(np.int)
    label = np.array(label)
    logger.info('%s, %s %s %s', label.shape, type(label), label.min(), label.max())
    return label


def predict_cv(x, y, train_ratio=0.20, splits=10, c=1.0, seed: Optional[int] = None):
    results = []
    shuffle = ShuffleSplit(n_splits=splits, test_size=1 - train_ratio, random_state=seed)
    for train_index, test_index in shuffle.split(x):
        logger.debug('%s, %s', train_index.shape, test_index.shape)
        assert len(set(train_index) & set(test_index)) == 0
        assert len(train_index) + len(test_index) == x.shape[0]
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = OneVsRestClassifier(
            LogisticRegression(
                C=c,
                solver="liblinear",
                multi_class="ovr",
            ),
            n_jobs=-1,
        )
        clf.fit(x_train, y_train)
        y_score = clf.predict_proba(x_test)
        y_pred = construct_indicator(y_score, y_test)
        mi = f1_score(y_test, y_pred, average="micro")
        ma = f1_score(y_test, y_pred, average="macro")
        try:
            roc_auc = roc_auc_score(y_test, y_score)
        except ValueError:
            logger.warning('Problem with roc_auc_score')
            continue

        logger.debug("micro f1 %.3f, macro f1 %.3f, roc-auc %.3f", mi, ma, roc_auc)
        results.append(dict(
            micro_f1=mi,
            macro_f1=ma,
            roc_auc=roc_auc,
        ))

    return results


def _get_embedding(embedding):
    logger.info("Loading network embedding from %s...", embedding)
    ext = os.path.splitext(embedding)[1]
    if ext == ".npy":
        return np.load(embedding)
    elif ext == ".pkl":
        with open(embedding, "rb") as f:
            return pkl.load(f)
    else:
        # Load word2vec format
        return load_w2v_feature(embedding)


def _help_main(*, label, matfile_variable_name, embedding, start_train_ratio, stop_train_ratio, num_train_ratio,
               splits, c, seed):
    logger.info("Loading label from %s...", label)
    label = load_label(matfile=label, variable_name=matfile_variable_name)
    logger.info("Label loaded!")

    train_ratios = np.linspace(start_train_ratio, stop_train_ratio, num_train_ratio)
    for train_ratio in train_ratios:
        results = predict_cv(embedding, label, train_ratio=train_ratio, splits=splits, c=c, seed=seed)

        micro_f1s, macro_f1s, roc_aucs = zip(*(
            (r['micro_f1'], r['macro_f1'], r['roc_auc'])
            for r in results
        ))
        logger.info("%d fold validation, training ratio %.3f", len(results), train_ratio)
        logger.info("Average micro %.3f, Average macro %.3f, Average ROC-AUC: %.3f",
                    np.mean(micro_f1s), np.mean(macro_f1s), np.mean(roc_aucs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, required=True,
                        help="input file path for labels (.mat)")
    parser.add_argument("--embedding", type=str, required=True,
                        help="input file path for embedding (.npy)")
    parser.add_argument("--matfile-variable-name", type=str, default='group',
                        help='variable name of adjacency matrix inside a .mat file.')
    parser.add_argument("--seed", type=int, required=False,
                        help="seed used for random number generator when randomly split data into training/test set.")
    parser.add_argument("--start-train-ratio", type=float, default=0.10,
                        help="the start value of the train ratio (inclusive).")
    parser.add_argument("--stop-train-ratio", type=float, default=0.90,
                        help="the end value of the train ratio (inclusive).")
    parser.add_argument("--num-train-ratio", type=int, default=9,
                        help="the number of train ratio chosen from [train-ratio-start, train-ratio-end].")
    parser.add_argument("--C", type=float, default=1.0,
                        help="inverse of regularization strength used in logistic regression.")
    parser.add_argument("--num-split", type=int, default=10,
                        help="The number of re-shuffling & splitting for each train ratio.")
    args = parser.parse_args()
    logging.basicConfig(
        # filename="%s.log" % args.embedding, filemode="w", # uncomment this to log to file
        level=logging.INFO,
        format='%(asctime)s %(message)s')  # include timestamp

    embedding = _get_embedding(args.embedding)
    logger.info("Network embedding loaded!")

    _help_main(
        label=args.label,
        matfile_variable_name=args.matfile_variable_name,
        embedding=embedding,
        start_train_ratio=args.start_train_ratio,
        stop_train_ratio=args.stop_train_ratio,
        num_train_ratio=args.num_train_ratio,
        splits=args.num_split,
        c=args.C,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
