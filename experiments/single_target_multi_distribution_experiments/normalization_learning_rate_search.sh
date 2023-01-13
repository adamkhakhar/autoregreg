GPU_IND=2
NUM_SAMPLES=200000
BATCH_SIZE=1000
NUM_WORKERS=2
LOG_EVERY=4
PROJECT_NAME=norm_lr_search

mkdir -p ../../results/result_logging/
echo SAME SCALE
for LEARNING_RATE in 10 1 .5 .1 .05 .01 .005 .001 .0005 .0001 .00005 .00001 .000005 .000001 
do
    echo LEARNING_RATE $LEARNING_RATE
    name=mnist_mse_norm_sin_s_log_s
    echo RUNNING $name
    python3 mnist_mse_mae_sin_log.py ${name} --normalization --wandb --wandb_project $PROJECT_NAME --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}.log &

    name=one_dim_mse_norm_sin_s_log_s
    echo RUNNING $name
    python3 one_dim_one_target_mse_mae_sin_log.py ${name} --normalization --wandb --wandb_project $PROJECT_NAME --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}.log &

    echo LEARNING_RATE $LEARNING_RATE
    name=mnist_mse_norm_sin_s_log_l
    echo RUNNING $name
    python3 mnist_mse_mae_sin_log.py ${name} --normalization --log l --wandb --wandb_project $PROJECT_NAME --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}.log &

    name=one_dim_mse_norm_sin_s_log_l
    echo RUNNING $name
    python3 one_dim_one_target_mse_mae_sin_log.py ${name} --normalization --log l --wandb --wandb_project $PROJECT_NAME --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}.log &
    wait
done