import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(ROOT + "/utils")
from create_plots import single_y_plot
import utils

MSE_COLOR = "darkred"
MAE_COLOR = "coral"
ARR_COLOR = "royalblue"
MAE_METRIC = "out_of_sample_error"
MSE_METRIC = "out_of_sample_mean_squared_error"
ARR_METRIC = "out_of_sample_hard_mean_squared_error"
labels = ["Mean Squared Error", "Mean Absolute Error", "Autoregressive"]
colors = [MSE_COLOR, MAE_COLOR, ARR_COLOR]

GENERATE_ONE_DIM_PLOTS = True
GENERATE_MNIST_PLOTS = True
GENERATE_LEGEND = True

if GENERATE_ONE_DIM_PLOTS:
    ONE_DIM_MAE_BATCH_SIZE = 10000
    one_dim_save_path = f"{CURR_DIR}/one_dim_plots"
    os.makedirs(one_dim_save_path, exist_ok=True)
    # one dim smaller targer error | log s sin l
    MSE_one_dim_error = utils.pull_from_s3(
        f"mse_log_s_sin_l_lr_0.0001_seed_1_mini_batch_metrics.bin"
    )[MSE_METRIC]
    MAE_one_dim_error = utils.pull_from_s3(
        f"mae_log_s_sin_l_lr_0.001_seed_1_mini_batch_metrics.bin"
    )[MAE_METRIC]
    for i in range(len(MAE_one_dim_error)):
        MAE_one_dim_error[i] = np.square(MAE_one_dim_error[i]) * ONE_DIM_MAE_BATCH_SIZE
    ARR_one_dim_error = utils.pull_from_s3(
        f"arr_log_s_sin_l_base_100_expmin_-3_expmax_4_lr_0.0001_seed_1_mini_batch_metrics.bin"
    )[ARR_METRIC]

    x = []
    y = []
    for l in [MSE_one_dim_error, MAE_one_dim_error, ARR_one_dim_error]:
        y.append(l[0])
        x.append([i for i in range(len(l[0]))])
    x_axis_label = "Gradient Steps (Thousands)"
    y_axis_label = "Log (Smaller Target): Error"
    single_y_plot(
        x,
        y,
        save_title="one_dim_small_log_s",
        save_path=one_dim_save_path,
        colors=colors,
        y_axis_exp=True,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
    )

    x = []
    y = []
    for l in [MSE_one_dim_error, MAE_one_dim_error, ARR_one_dim_error]:
        y.append([l[0][i] + l[1][i] for i in range(len(l[0]))])
        x.append([i for i in range(len(l[0]))])
    x_axis_label = "Gradient Steps (Thousands)"
    y_axis_label = "Combined Error"
    single_y_plot(
        x,
        y,
        save_title="one_dim_log_s_sin_l",
        save_path=one_dim_save_path,
        colors=colors,
        y_axis_exp=True,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
    )

    # one dim smaller targer error | sin s log l
    MSE_one_dim_error = utils.pull_from_s3(
        f"mse_sin_s_log_l_lr_0.0001_seed_1_mini_batch_metrics.bin"
    )[MSE_METRIC]
    MAE_one_dim_error = utils.pull_from_s3(
        f"mae_sin_s_log_l_lr_0.001_seed_1_mini_batch_metrics.bin"
    )[MAE_METRIC]
    for i in range(len(MAE_one_dim_error)):
        MAE_one_dim_error[i] = np.square(MAE_one_dim_error[i]) * ONE_DIM_MAE_BATCH_SIZE
    ARR_one_dim_error = utils.pull_from_s3(
        f"arr_sin_s_log_l_base_100_expmin_-3_expmax_4_lr_0.0001_seed_1_mini_batch_metrics.bin"
    )[ARR_METRIC]

    x = []
    y = []
    for l in [MSE_one_dim_error, MAE_one_dim_error, ARR_one_dim_error]:
        y.append(l[0])
        x.append([i for i in range(len(l[0]))])
    x_axis_label = "Gradient Steps (Thousands)"
    y_axis_label = "Sin (Smaller Target): Error"
    single_y_plot(
        x,
        y,
        save_title="one_dim_small_sin_s",
        save_path=one_dim_save_path,
        colors=colors,
        y_axis_exp=True,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
    )

    x = []
    y = []
    for l in [MSE_one_dim_error, MAE_one_dim_error, ARR_one_dim_error]:
        y.append([l[0][i] + l[1][i] for i in range(len(l[0]))])
        x.append([i for i in range(len(l[0]))])
    x_axis_label = "Gradient Steps (Thousands)"
    y_axis_label = "Combined Error"
    single_y_plot(
        x,
        y,
        save_title="one_dim_sin_s_log_l",
        save_path=one_dim_save_path,
        colors=colors,
        y_axis_exp=True,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
    )

    # one dim smaller targer error | sin s log l
    MSE_one_dim_error = utils.pull_from_s3(
        f"mse_sin_s_log_s_lr_1e-05_seed_1_mini_batch_metrics.bin"
    )[MSE_METRIC]
    MAE_one_dim_error = utils.pull_from_s3(
        f"mae_sin_s_log_s_lr_0.005_seed_1_mini_batch_metrics.bin"
    )[MAE_METRIC]
    for i in range(len(MAE_one_dim_error)):
        MAE_one_dim_error[i] = np.square(MAE_one_dim_error[i]) * ONE_DIM_MAE_BATCH_SIZE
    ARR_one_dim_error = utils.pull_from_s3(
        f"arr_sin_s_log_s_base_100_expmin_-3_expmax_4_lr_0.0001_seed_1_mini_batch_metrics.bin"
    )[ARR_METRIC]

    x = []
    y = []
    for l in [MSE_one_dim_error, MAE_one_dim_error, ARR_one_dim_error]:
        y.append(l[0])
        x.append([i for i in range(len(l[0]))])
    x_axis_label = "Gradient Steps (Thousands)"
    y_axis_label = "Sin: Error"
    single_y_plot(
        x,
        y,
        save_title="one_dim_both_small_sin_s",
        save_path=one_dim_save_path,
        colors=colors,
        y_axis_exp=True,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
    )

    x = []
    y = []
    for l in [MSE_one_dim_error, MAE_one_dim_error, ARR_one_dim_error]:
        y.append(l[1])
        x.append([i for i in range(len(l[0]))])
    x_axis_label = "Gradient Steps (Thousands)"
    y_axis_label = "Log: Error"
    single_y_plot(
        x,
        y,
        save_title="one_dim_both_small_log_s",
        save_path=one_dim_save_path,
        colors=colors,
        y_axis_exp=True,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
    )

if GENERATE_MNIST_PLOTS:
    MNIST_MAE_BATCH_SIZE = 1000
    mnist_save_path = f"{CURR_DIR}/mnist_plots"
    os.makedirs(mnist_save_path, exist_ok=True)
    # mnist smaller targer error | log s sin l
    MSE_mnist_error = utils.pull_from_s3(
        f"mnist_mse_log_s_sin_l_lr_0.005_seed_1_mini_batch_metrics.bin"
    )[MSE_METRIC]
    MAE_mnist_error = utils.pull_from_s3(
        f"mnist_mae_log_s_sin_l_lr_0.005_seed_1_mini_batch_metrics.bin"
    )[MAE_METRIC]
    for i in range(len(MAE_mnist_error)):
        MAE_mnist_error[i] = np.square(MAE_mnist_error[i]) * MNIST_MAE_BATCH_SIZE
    ARR_mnist_error = utils.pull_from_s3(
        f"mnist_arr_log_s_sin_l_base_100_expmin_-3_expmax_4_lr_0.0005_seed_10_mini_batch_metrics.bin"
    )[ARR_METRIC]

    x = []
    y = []
    for l in [MSE_mnist_error, MAE_mnist_error, ARR_mnist_error]:
        y.append(l[0])
        x.append([i for i in range(len(l[0]))])
    x_axis_label = "Gradient Steps (Thousands)"
    y_axis_label = "Log (Smaller Target): Error"
    single_y_plot(
        x,
        y,
        save_title="mnist_small_log_s",
        save_path=mnist_save_path,
        colors=colors,
        y_axis_exp=True,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
    )

    x = []
    y = []
    for l in [MSE_mnist_error, MAE_mnist_error, ARR_mnist_error]:
        y.append([l[0][i] + l[1][i] for i in range(len(l[0]))])
        x.append([i for i in range(len(l[0]))])
    x_axis_label = "Gradient Steps (Thousands)"
    y_axis_label = "Combined Error"
    single_y_plot(
        x,
        y,
        save_title="mnist_log_s_sin_l",
        save_path=mnist_save_path,
        colors=colors,
        y_axis_exp=True,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
    )
    del MSE_mnist_error
    del MAE_mnist_error
    del ARR_mnist_error

    # mnist smaller targer error | sin s log l
    MSE_mnist_error = utils.pull_from_s3(
        f"mnist_mse_sin_s_log_l_lr_0.005_seed_1_mini_batch_metrics.bin"
    )[MSE_METRIC]
    MAE_mnist_error = utils.pull_from_s3(
        f"mnist_mae_sin_s_log_l_lr_0.005_seed_1_mini_batch_metrics.bin"
    )[MAE_METRIC]
    for i in range(len(MAE_mnist_error)):
        MAE_mnist_error[i] = np.square(MAE_mnist_error[i]) * MNIST_MAE_BATCH_SIZE
    ARR_mnist_error = utils.pull_from_s3(
        f"mnist_arr_sin_s_log_l_base_100_expmin_-3_expmax_4_lr_0.0005_seed_10_mini_batch_metrics.bin"
    )[ARR_METRIC]

    x = []
    y = []
    for l in [MSE_mnist_error, MAE_mnist_error, ARR_mnist_error]:
        y.append(l[0])
        x.append([i for i in range(len(l[0]))])
    x_axis_label = "Gradient Steps (Thousands)"
    y_axis_label = "Sin (Smaller Target): Error"
    single_y_plot(
        x,
        y,
        save_title="mnist_small_sin_s",
        save_path=mnist_save_path,
        colors=colors,
        y_axis_exp=True,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
    )

    x = []
    y = []
    for l in [MSE_mnist_error, MAE_mnist_error, ARR_mnist_error]:
        y.append([l[0][i] + l[1][i] for i in range(len(l[0]))])
        x.append([i for i in range(len(l[0]))])
    x_axis_label = "Gradient Steps (Thousands)"
    y_axis_label = "Combined Error"
    single_y_plot(
        x,
        y,
        save_title="mnist_sin_s_log_l",
        save_path=mnist_save_path,
        colors=colors,
        y_axis_exp=True,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
    )

    del MSE_mnist_error
    del MAE_mnist_error
    del ARR_mnist_error
    # mnist smaller targer error | sin s log s
    MSE_mnist_error = utils.pull_from_s3(
        f"mnist_mse_sin_s_log_s_lr_0.0001_seed_5_mini_batch_metrics.bin"
    )[MSE_METRIC]
    MAE_mnist_error = utils.pull_from_s3(
        f"mnist_mae_sin_s_log_s_lr_0.0001_seed_1_mini_batch_metrics.bin"
    )[MAE_METRIC]
    for i in range(len(MAE_mnist_error)):
        MAE_mnist_error[i] = np.square(MAE_mnist_error[i]) * MNIST_MAE_BATCH_SIZE
    ARR_mnist_error = utils.pull_from_s3(
        f"mnist_arr_sin_s_log_s_base_100_expmin_-3_expmax_4_lr_0.0005_seed_5_mini_batch_metrics.bin"
    )[ARR_METRIC]

    x = []
    y = []
    for l in [MSE_mnist_error, MAE_mnist_error, ARR_mnist_error]:
        y.append(l[0])
        x.append([i for i in range(len(l[0]))])
    x_axis_label = "Gradient Steps (Thousands)"
    y_axis_label = "Sin: Error"
    single_y_plot(
        x,
        y,
        save_title="mnist_both_small_sin_s",
        save_path=mnist_save_path,
        colors=colors,
        y_axis_exp=True,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
    )

    x = []
    y = []
    for l in [MSE_mnist_error, MAE_mnist_error, ARR_mnist_error]:
        y.append(l[1])
        x.append([i for i in range(len(l[0]))])
    x_axis_label = "Gradient Steps (Thousands)"
    y_axis_label = "Log: Error"
    single_y_plot(
        x,
        y,
        save_title="mnist_both_small_log_s",
        save_path=mnist_save_path,
        colors=colors,
        y_axis_exp=True,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
    )

if GENERATE_LEGEND:
    cs = [patches.Patch(facecolor=colors[i]) for i in range(len(colors))]
    plt.legend(cs, labels)
    plt.savefig(f"{CURR_DIR}/legend.png", bbox_inches="tight", dpi=2000)
