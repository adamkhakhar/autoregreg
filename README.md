# Autoregressive Regression
Adam Khakhar and Jacob Buckman
(Insert intro and link paper.)

# Quick Start Guide
(Insert quick start guide.)

# Codebase Tree
```
.
├── experiments
│   ├── launch_scripts
│   │   ├── mnist_arr_launch_script.sh
│   │   ├── mnist_arr.log
│   │   ├── mnist_arr_lr_search.sh
│   │   ├── mnist_mae_launch_script.sh
│   │   ├── mnist_mae_lr_search.sh
│   │   ├── mnist_mse_launch_script.sh
│   │   ├── mnist_mse.log
│   │   ├── mnist_mse_lr_search.sh
│   │   ├── one_dim_arr_launch_script.sh
│   │   ├── one_dim_mae_lr_search.sh
│   │   └── one_dim_mse_launch_script.sh
│   ├── mnist_arr.py
│   ├── mnist_experiments
│   │   ├── mnist_arr_log_s_sin_l.py
│   │   ├── mnist_arr_sin_s_log_l.py
│   │   ├── mnist_arr_sin_s_log_s.py
│   │   ├── mnist_mae_log_s_sin_l.py
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
│   └── one_dim_mse.py
├── LICENSE
├── README.md
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
│       ├── MAETrain.py
│       ├── MSETrain.py
│       └── Train.py
└── utils
    ├── target_functions.py
    └── utils.py
```
