# Based on evaluate_zeroshot from SLIP but changed by MB
from __future__ import annotations

import copy
import random
import time
from collections import defaultdict

import torch
import torch.utils.data
from loguru import logger

from .logistic_regression import logreg_linear, logreg_mha
from .metrics import Accuracy


def evaluate_sgd_probe(
    train_feats,
    train_labels,
    valid_feats,
    valid_labels,
    test_feats,
    test_labels,
    use_mean_accuracy,
):
    """
    Args:
        holdout_fraction: Fraction of the (official) train split to hold out for
            validation while searching the cost hyperparameter. Holding out will
            be deterministic and have similar class distribution as train split.
    """
    start = time.time()
    classifier = train_sgd_probe(
        train_feats, train_labels, valid_feats, valid_labels, use_mean_accuracy
    )
    test_acc = test_sgd_probe(classifier, test_feats, test_labels, use_mean_accuracy)

    del classifier
    torch.cuda.empty_cache()
    print(f"Time taken {time.time() - start:.2f}")
    return test_acc


def train_sgd_probe(
    train_feats, train_labels, valid_feats, valid_labels, use_mean_accuracy
):
    """
    Args:
        holdout_fraction: Fraction of the (official) train split to hold out for
            validation while searching the cost hyperparameter. Holding out will
            be deterministic and have similar class distribution as train split.
    """
    NUM_C = len(set(train_labels.cpu().numpy()))
    acc_meter = Accuracy(num_classes=NUM_C, mean_per_class=use_mean_accuracy)

    if valid_labels is None:
        trainval_feats = train_feats
        trainval_labels = train_labels
        train_ind, valid_ind = split_trainval(trainval_labels.cpu().numpy(), 0.2)
        train_feats = trainval_feats[train_ind]
        train_labels = trainval_labels[train_ind]
        valid_feats = trainval_feats[valid_ind]
        valid_labels = trainval_labels[valid_ind]

    n_epoch = 100
    if True:
        classifier = logreg_mha(train_feats.shape[-1], NUM_C)
    else:
        classifier = logreg_linear(train_feats.shape[-1], NUM_C)

    final_classifier = copy.deepcopy(classifier)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_acc = -1

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.01)

    for epoch in range(n_epoch):
        optimizer.zero_grad()
        train_pred = classifier(train_feats)
        loss = loss_fn(train_pred, train_labels)
        loss.backward()
        optimizer.step()

        val_pred = classifier(valid_feats)
        val_acc = acc_meter(val_pred, valid_labels)
        trn_acc = acc_meter(train_pred, train_labels)

        acc_print = f"train_acc: {trn_acc:.2f} | val acc: {val_acc:.2f}"
        print(f"{epoch:3d}/{n_epoch:3d} | loss: {loss.item():.2f} | {acc_print}")
        if val_acc > best_acc:
            best_acc = val_acc
            final_classifier = copy.deepcopy(classifier)

    return final_classifier


def test_sgd_probe(
    linear_classifier, test_feats, test_labels, use_mean_accuracy, num_classes=None
):
    # evaluate
    NUM_C = len(set(test_labels.cpu().numpy())) if num_classes is None else num_classes
    acc_meter = Accuracy(num_classes=NUM_C, mean_per_class=use_mean_accuracy)
    predictions = linear_classifier(test_feats).softmax(dim=-1)
    accuracy = acc_meter(torch.as_tensor(predictions), test_labels)

    logger.info(f"Test accuracy: {accuracy:.3f}")
    return accuracy


def split_trainval(targets, val_percentage):
    # Organize dataset by classes (class ID -> list[dataset index] map).
    labels_to_indices = defaultdict(list)
    for index, label in enumerate(targets):
        labels_to_indices[label].append(index)

    train_indices = []
    valid_indices = []
    for label, indices in labels_to_indices.items():
        # Deterministic shuffling to ensure same held-out split across runs.
        random.Random(93).shuffle(indices)

        train_indices.extend(indices[int(len(indices) * val_percentage) :])
        valid_indices.extend(indices[: int(len(indices) * val_percentage)])

    return train_indices, valid_indices
