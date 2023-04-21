# Evaluating the representations learned by SSL methods

The aim of project is to provide a comprehensive evaluation of the features learned by SSL methods. 
At its core, the method would extract pre-trained features and combine them with a zero-shot
inference procedure or a (non-)linear probe for a variety of tasks. 
This extends the existing work by Kornblith on providing an evaluation scheme for this. 

**To do**
- Get evaluation with existing repo 
- Get some major pre-trained backbones into the repo and extract dense/global features from them
    - Question: which methods have a [CLS] 
    - [ ] DINO
    - [ ] DINO v2
    - [ ] CLIP
    - [ ] MAE
    - [ ] ODISE-style feature extraction from StableDiffusion
    - [ ] ViT 




Environment Setup
-----------------

We recommend using Anaconda or Miniconda. To setup the environment, follow the instructions below.

```bash
conda create -n ssl_evals python=3.9 --yes
conda activate ssl_evals
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia

python -m pip install -r requirements.txt
python setup.py develop

# weird dependency with datasets and google's api
python -m pip install protobuf==3.20.3 
```

Evaluation Datasets
-------------------

We use [TensorFlow Datasets](https://www.tensorflow.org/datasets) for our evaluations. 
This package provides us with all the evaluations except for FGVC Aircraft. 
Our code will automatically download and extract all the
datasets in `data/evaluation_datasets` on the first run of the evaluation code.
This means that the first evaluation run will be much slower than usual.  

**Note 1:** We encountered a bug with SUN 397 where one image could not be decoded correctly. 
This is a known [bug](https://github.com/tensorflow/datasets/pull/3951) which has not been fixed yet
in the stable version. To fix it, simply make the two changes outlined by this
[commit](https://github.com/tensorflow/datasets/pull/3951/commits/c4ff599a357ee92f5f0584efb715939299f1d13e).

**Note 2:**
TensorFlow Datasets will require you to independently downloaded RESISC45. Please follow the
instructions provided [here](https://www.tensorflow.org/datasets/catalog/resisc45)


Evaluation
-----------

We use two primary evaluations: linear probe using L-BFGS and few-shot evaluation. The configs for those evaluations can be found [here](./configs/evaluation.yaml). 

**Linear Probe**: we train a single layer using logistic regression and sweep over regularizer weight values. 
We provide an [implementation](./lgssl/evaluation/logistic_regression.py) of logistic regression using PyTorch's L-BFGS, however, you can easily use scikit-learn's implementation by setting the `use_sklearn` flag in the [evaluation configs](./configs/evaluation.yaml).
For datasets without a standard validation split, we randomly split the training set while maintaining the class distribution. 

**Few-Shot Evaluation**: we also evaluate our frozen features on 5-shot, 5-way classification. The evaluation can be found [here](./lgssl/evaluation/fewshot.py).
We sample the training samples from the train/valid splits and the query samples for the test set. 

The following commands can be used to evaluate checkpoints or baselines. TO DO
