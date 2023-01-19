# Deep Autoregressive Regression
Adam Khakhar and Jacob Buckman\
[Paper](https://arxiv.org/abs/2211.07447)\
Citation Information:
```
@misc{https://doi.org/10.48550/arxiv.2211.07447,
  doi = {10.48550/ARXIV.2211.07447},
  
  url = {https://arxiv.org/abs/2211.07447},
  
  author = {Khakhar, Adam and Buckman, Jacob},
  
  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Deep Autoregressive Regression},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```


# Quick Start Guide
### Start Training
See examples of training on datasets with various model configurations and training objectives in the experiments directory. At a high level, training includes instatiating a dataset, model, and training class.

### Datasets
DataSet impelementations for one-dimensional functions, MNIST, and the Amazon Review DataSet are in `src/datasets`.

### Models
Feed Forward, Transformer, and CNN models are implemented. Also implemented is the AutoRegressive Head, implementing the paper's main contribution, autoregressive regression. These models in `src/models` can be combined in an Encoder-Decoder framework using the EncoderDecoder class.

### Training
Various training objectives can be used to train the models. Additionally, different error metrics can be computed at each gradient update step. The set of classes in the `src/training` implement training with the following objectives: mean squared error, mean absolute error, and autoregressive regression.

# Codebase Tree
```
.
├── experiments
│   └── single_target_multi_distribution_experiments
│       ├── mnist_arr_sin_log.py
│       ├── mnist_mse_mae_sin_log.py
│       ├── one_dim_one_target_arr_sin_log.py
│       └── one_dim_one_target_mse_mae_sin_log.py
├── LICENSE
├── README.md
├── src
│   ├── datasets
│   │   ├── MNISTDataSet.py
│   │   ├── MNISTMultiDistributionSingleTarget.py
│   │   ├── OneDimDataSet.py
│   │   ├── OneDimMultiDistributionSingleTarget.py
│   │   ├── ReviewDataSet.py
│   │   └── TestTransformerDataSet.py
│   ├── models
│   │   ├── AutoRegressiveHead.py
│   │   ├── CNN.py
│   │   ├── EncoderDecoder.py
│   │   ├── FeedForward.py
│   │   └── Transformer.py
│   └── training
│       ├── ARRTrain.py
│       ├── MAETrain.py
│       ├── MSETrain.py
│       └── Train.py
└── utils
    ├── create_plots.py
    ├── PositionalEncoding.py
    ├── target_functions.py
    └── utils.py
```
