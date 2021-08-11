# CF-GNNExplainer: Counterfactual Explanations for Graph Neural Networks

This repository is the official implementation of the paper CF-GNNExplainer: Counterfactual Explanations for Graph Neural Networks. 


## Requirements

To install requirements:

```setup
conda env create --file environment.yml
```

>📋 This will create a conda environment called pytorch-geo

## Training original models

To train the original GNN models for the BA-shapes dataset in the paper, cd into src and run this command:

```train
python train.py --dataset=syn1
```

>📋  For the Tree-Cycles dataset, the dataset argument should be "syn4". For the Tree-Grid dataset, it should be "syn5". All hyperparameter settings are listed in the defaults, and all models have the same hyperparameters. 


## Training CF-GNNExplainer

To train CF-GNNExplainer for each dataset, run the following commands:

```train
python train.py --dataset=syn1 --lr=0.01 --beta=0.5 --n_momentum=0.9 --optimizer=SGD
python train.py --dataset=syn4 --lr=0.1 --beta=0.5 --optimizer=SGD
python train.py --dataset=syn5 --lr=0.1 --beta=0.5 --optimizer=SGD
```

>📋  This will create another folder in the main directory called 'results', where the results files will be stored.


## Evaluation

To evaluate the CF examples, loop over all the files in the results folder by running the following command:

```eval
python evaluate.py
```
>📋  This will print out the values for each metric, for each results file.

## Pre-trained Models

The pretrained models are available in the models folder


## Results

Our model achieves the following performance:

| Model name         | Dataset        | Fidelity       |  Size |    Sparsity   | Accuracy    |
| ------------------ |---------------- | -------------- | -------------- | -------------- |   -------------- |
| CF-GNNExplainer   |     Tree-Cycles  |      0.21       |      2.09           |       0.90        |      0.94       |
| CF-GNNExplainer   |     Tree-Grid    |      0.07       |       1.47          |      0.94         |     0.96        |
| CF-GNNExplainer   |     BA-Shapes    |      0.39       |       2.39          |       0.99        |      0.96        |
