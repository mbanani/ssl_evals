from __future__ import annotations

import hydra
from loguru import logger
from omegaconf import DictConfig

from evals.datasets.builder import get_imagenet_loaders
from evals.evaluation.fewshot import evaluate_fewshot
from evals.evaluation.linear_probe import evaluate_linear_probe
from evals.evaluation.utils import extract_features, get_model


@hydra.main(config_name="evaluation", config_path="./configs", version_base=None)
def main(cfg: DictConfig):
    model = get_model(cfg.model.name, cfg.model.checkpoint)
    assert any([cfg.evaluations[x] for x in cfg.evaluations]), "No evaluation tasks."

    logger.info(f"Evaluating {cfg.model.name} - {cfg.model.checkpoint}")
    use_mean_acc = False

    train_loader, valid_loader = get_imagenet_loaders(
        cfg.dataset.image_mean, image_size=cfg.dataset.image_size, small=False
    )

    # extract features
    train_feats, train_labels = extract_features(model, train_loader)
    valid_feats, valid_labels = extract_features(model, valid_loader)
    num_classes = len(valid_labels.unique())

    num_train = len(train_loader.dataset)
    num_valid = len(valid_loader.dataset)

    logger.info(f"imagenet | train: {num_train} val: {num_valid}")
    logger.info(f"imagenet | num classes: {num_classes}")

    if cfg.evaluations.linear_probe:
        lin_out = evaluate_linear_probe(
            train_feats,
            train_labels,
            None,
            None,
            valid_feats,
            valid_labels,
            use_mean_acc,
            max_iter=cfg.logistic_regression.max_iter,
            combine_trainval=cfg.logistic_regression.combine_trainval,
            use_sklearn=cfg.logistic_regression.use_sklearn,
        )
    else:
        lin_out = -1

    if cfg.evaluations.fewshot:
        few_out = evaluate_fewshot(
            train_feats, train_labels, valid_feats, valid_labels, cfg.fewshot
        )
    else:
        few_out = (-1, -1)

    logger.info(f"Model: {cfg.model.name} || {cfg.model.checkpoint}")
    logger.info(f"linear probe on imagenet: {lin_out:.2f}")
    logger.info(f"fewshot on imagenet: {few_out[0]:.2f} +/- {few_out[1]:.2f}")


if __name__ == "__main__":
    main()
