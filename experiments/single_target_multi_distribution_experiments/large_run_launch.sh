# NUM_SAMPLES=10000000
# BATCH_SIZE=1000
# LOG_EVERY=100
NUM_SAMPLES=100
BATCH_SIZE=10
LOG_EVERY=1
###
BASE=100
EXP_MIN=-3
EXP_MAX=4
NUM_WORKERS=2

GPU_IND=3
################################################################################################
#ONE DIM SIN S, LOG S
project=one_dim_sin_s_log_s
#ARR
name=arr
LEARNING_RATE=.005
python3 one_dim_one_target_arr_sin_log.py ${name} --wandb --wandb_project $project --base $BASE --exp_min $EXP_MIN --exp_max $EXP_MAX --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE  --num_samples_error_track $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}_${project}.log &

#MSE Norm
name=mse_norm
LEARNING_RATE=.001
python3 one_dim_one_target_mse_mae_sin_log.py ${name} --normalization --wandb --wandb_project $project --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}_${project}.log &

#MSE
name=mse
LEARNING_RATE=.05
python3 one_dim_one_target_mse_mae_sin_log.py ${name} --wandb --wandb_project $project --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}_${project}.log &

#MAE
name=mae
LEARNING_RATE=.0005
python3 one_dim_one_target_mse_mae_sin_log.py ${name} --mae --wandb --wandb_project $project --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}_${project}.log &

GPU_IND=4
################################################################################################
#ONE DIM SIN S, LOG L
project=one_dim_sin_s_log_l
#ARR
name=arr
LEARNING_RATE=.005
python3 one_dim_one_target_arr_sin_log.py ${name} --log l --wandb --wandb_project $project --base $BASE --exp_min $EXP_MIN --exp_max $EXP_MAX --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_samples_error_track $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}_${project}.log &

#MSE Norm
name=mse_norm
LEARNING_RATE=.001
python3 one_dim_one_target_mse_mae_sin_log.py ${name} --log l --normalization --wandb --wandb_project $project --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}_${project}.log &

#MSE
name=mse
LEARNING_RATE=.5
python3 one_dim_one_target_mse_mae_sin_log.py ${name} --log l --wandb --wandb_project $project --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}_${project}.log &

#MAE
name=mae
LEARNING_RATE=.5
python3 one_dim_one_target_mse_mae_sin_log.py ${name} --mae --log l --wandb --wandb_project $project --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}_${project}.log &

wait
GPU_IND=3
################################################################################################
#ONE DIM SIN L, LOG S
project=one_dim_sin_l_log_s
#ARR
name=arr
LEARNING_RATE=.005
python3 one_dim_one_target_arr_sin_log.py ${name} --sin l --wandb --wandb_project $project --base $BASE --exp_min $EXP_MIN --exp_max $EXP_MAX --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_samples_error_track $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}_${project}.log &

#MSE Norm
name=mse_norm
LEARNING_RATE=.001
python3 one_dim_one_target_mse_mae_sin_log.py ${name} --sin l --normalization --wandb --wandb_project $project --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}_${project}.log &

#MSE
name=mse
LEARNING_RATE=.5
python3 one_dim_one_target_mse_mae_sin_log.py ${name} --sin l --wandb --wandb_project $project --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}_${project}.log &

#MAE
name=mae
LEARNING_RATE=.5
python3 one_dim_one_target_mse_mae_sin_log.py ${name} --mae --sin l --wandb --wandb_project $project --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}_${project}.log &

GPU_IND=4
################################################################################################
#MNIST SIN S, LOG S
project=mnist_sin_s_log_s
#ARR
name=arr
LEARNING_RATE=.0005
python3 mnist_arr_sin_log.py ${name} --wandb --wandb_project $project --base $BASE --exp_min $EXP_MIN --exp_max $EXP_MAX --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE  --num_samples_error_track $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}_${project}.log &

#MSE Norm
name=mse_norm
LEARNING_RATE=.0005
python3 mnist_mse_mae_sin_log.py ${name} --normalization --wandb --wandb_project $project --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}_${project}.log &

#MSE
name=mse
LEARNING_RATE=.01
python3 mnist_mse_mae_sin_log.py ${name} --wandb --wandb_project $project --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}_${project}.log &

#MAE
name=mae
LEARNING_RATE=.01
python3 mnist_mse_mae_sin_log.py ${name} --mae --wandb --wandb_project $project --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}_${project}.log &

wait
GPU_IND=3
################################################################################################
#MNIST SIN S, LOG L
project=mnist_sin_s_log_l
#ARR
name=arr
LEARNING_RATE=.0005
python3 mnist_arr_sin_log.py ${name} --log l --wandb --wandb_project $project --base $BASE --exp_min $EXP_MIN --exp_max $EXP_MAX --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_samples_error_track $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}_${project}.log &

#MSE Norm
name=mse_norm
LEARNING_RATE=.0005
python3 mnist_mse_mae_sin_log.py ${name} --log l --normalization --wandb --wandb_project $project --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}_${project}.log &

#MSE
name=mse
LEARNING_RATE=.0005
python3 mnist_mse_mae_sin_log.py ${name} --log l --wandb --wandb_project $project --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}_${project}.log &

#MAE
name=mae
LEARNING_RATE=.0005
python3 mnist_mse_mae_sin_log.py ${name} --mae --log l --wandb --wandb_project $project --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}_${project}.log &


GPU_IND=4
################################################################################################
#MNIST SIN L, LOG S
project=mnist_sin_l_log_s
#ARR
name=arr
LEARNING_RATE=.0005
python3 mnist_arr_sin_log.py ${name} --sin l --wandb --wandb_project $project --base $BASE --exp_min $EXP_MIN --exp_max $EXP_MAX --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_samples_error_track $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}_${project}.log &

#MSE Norm
name=mse_norm
LEARNING_RATE=.0005
python3 mnist_mse_mae_sin_log.py ${name} --sin l --normalization --wandb --wandb_project $project --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}_${project}.log &

#MSE
name=mse
LEARNING_RATE=.0005
python3 mnist_mse_mae_sin_log.py ${name} --sin l --wandb --wandb_project $project --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}_${project}.log &

#MAE
name=mae
LEARNING_RATE=.0005
python3 mnist_mse_mae_sin_log.py ${name} --mae --sin l --wandb --wandb_project $project --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}_${project}.log &
