# Deep Autoregressive Regression
Adam Khakhar and Jacob Buckman
[Paper](https://arxiv.org/abs/2211.07447)
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
│   ├── amazon_review_experiments
│   │   ├── amzn_arr.py
│   │   └── amzn_mse_mae.py
│   ├── launch_scripts
│   │   ├── amzn_review_arr_lr_search.sh
│   │   ├── amzn_review_mse_mae_launch_script.sh
│   │   ├── amzn_review_mse_mae_lr_search.sh
│   │   ├── mnist_arr_launch_script.sh
│   │   ├── mnist_arr_lr_search.sh
│   │   ├── mnist_mae_launch_script.sh
│   │   ├── mnist_mae_lr_search.sh
│   │   ├── mnist_mae_lr_sensivity_launch_script.sh
│   │   ├── mnist_mse_launch_script.sh
│   │   ├── mnist_mse_lr_search.sh
│   │   ├── one_dim_arr_launch_script.sh
│   │   ├── one_dim_mae_launch_script.sh
│   │   ├── one_dim_mae_lr_search.sh
│   │   └── one_dim_mse_launch_script.sh
│   ├── mnist_arr.py
│   ├── mnist_experiments
│   │   ├── mnist_arr_log_s_sin_l.py
│   │   ├── mnist_arr_sin_s_log_l.py
│   │   ├── mnist_arr_sin_s_log_s.py
│   │   ├── mnist_mae_log_s_sin_l.py
│   │   ├── mnist_mae_lr_sensitivity.py
│   │   ├── mnist_mae_sin_s_log_l.py
│   │   ├── mnist_mae_sin_s_log_s.py
│   │   ├── mnist_mse_log_s_sin_l.py
│   │   ├── mnist_mse_sin_s_log_l.py
│   │   └── mnist_mse_sin_s_log_s.py
│   ├── mnist_mae.py
│   ├── mnist_mse.py
│   ├── one_dim_arr.py
│   ├── one_dim_experiments
│   │   ├── one_dim_arr_log_s_sin_l.py
│   │   ├── one_dim_arr_sin_s_log_l.py
│   │   ├── one_dim_arr_sin_s_log_s.py
│   │   ├── one_dim_mae_log_s_sin_l.py
│   │   ├── one_dim_mae_sin_s_log_l.py
│   │   ├── one_dim_mae_sin_s_log_s.py
│   │   ├── one_dim_mse_log_s_sin_l.py
│   │   ├── one_dim_mse_sin_s_log_l.py
│   │   └── one_dim_mse_sin_s_log_s.py
│   ├── one_dim_mse.py
│   └── test_transformer.py
├── LICENSE
├── README.md
├── results
│   ├── monitor_training.ipynb
│   └── result_logging
├── src
│   ├── datasets
│   │   ├── MNISTDataSet.py
│   │   ├── OneDimDataSet.py
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
