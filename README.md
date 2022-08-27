# Autoregressive Regression
Adam Khakhar and Jacob Buckman
(Insert intro and link paper.)

# Quick Start Guide
(Insert quick start guide.)

# Codebase Tree
```
.
├── LICENSE
├── README.md
├── experiments
│   ├── mnist_arr.py
│   ├── mnist_mse.py
│   ├── one_dim_arr.py
│   └── one_dim_mse.py
├── launch_scripts
├── results
│   ├── monitor_training.ipynb
│   ├── test_experiment_data.bin
│   └── test_mini_batch_metrics.bin
├── src
│   ├── datasets
│   │   ├── MNISTDataSet.py
│   │   ├── OneDimDataSet.py
│   │   └── ReviewDataSet.py
│   ├── models
│   │   ├── AutoRegressiveHead.py
│   │   ├── CNN.py
│   │   ├── EncoderDecoder.py
│   │   └── FeedForward.py
│   └── training
│       ├── ARRTrain.py
│       ├── MSETrain.py
│       └── Train.py
└── utils
    └── utils.py
```
