GPU_IND=4
NUM_SAMPLES=100000
BATCH_SIZE=1000
NUM_WORKERS=4
LOG_EVERY=10
BASE=100
EXP_MIN=-3
EXP_MAX=4
ARR_TIMEOUT=3600
M_TIMEOUT=1800
PROJECT_NAME=mnist_lr_search

mkdir -p ../../results/result_logging/
echo SAME SCALE
for LEARNING_RATE in 10 1 .5 .1 .05 .01 .005 .001 .0005 .0001 .00005 .00001 .000005 .000001 
do
    echo LEARNING_RATE $LEARNING_RATE
    name=mnist_arr_sin_s_log_s
    echo RUNNING $name
    timeout $ARR_TIMEOUT python3 mnist_arr_sin_log.py ${name} --wandb --wandb_project $PROJECT_NAME  --num_samples_error_track $BATCH_SIZE --base $BASE --exp_min $EXP_MIN --exp_max $EXP_MAX --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}.log &

    echo LEARNING_RATE $LEARNING_RATE
    name=mnist_mse_sin_s_log_s
    echo RUNNING $name
    timeout $M_TIMEOUT python3 mnist_mse_mae_sin_log.py ${name} --wandb --wandb_project $PROJECT_NAME --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}.log &

    echo LEARNING_RATE $LEARNING_RATE
    name=mnist_mae_sin_s_log_s
    echo RUNNING $name
    timeout $M_TIMEOUT python3 mnist_mse_mae_sin_log.py ${name} --wandb --mae --wandb_project $PROJECT_NAME --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}.log &
    wait
done

echo DIFFERENT_SCALE
for LEARNING_RATE in 10 1 .5 .1 .05 .01 .005 .001 .0005 .0001 .00005 .00001 .000005 .000001 
do
    echo LEARNING_RATE $LEARNING_RATE
    name=mnist_arr_sin_l_log_s
    echo RUNNING $name
    timeout $ARR_TIMEOUT python3 mnist_arr_sin_log.py ${name} --sin l --wandb --wandb_project $PROJECT_NAME  --num_samples_error_track $BATCH_SIZE --base $BASE --exp_min $EXP_MIN --exp_max $EXP_MAX --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}.log &

    echo LEARNING_RATE $LEARNING_RATE
    name=mnist_mse_sin_l_log_s
    echo RUNNING $name
    timeout $M_TIMEOUT python3 mnist_mse_mae_sin_log.py ${name} --sin l --wandb --wandb_project $PROJECT_NAME  --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}.log &

    echo LEARNING_RATE $LEARNING_RATE
    name=mnist_mae_sin_l_log_s
    echo RUNNING $name
    timeout $M_TIMEOUT python3 mnist_mse_mae_sin_log.py ${name} --sin l --wandb --wandb_project $PROJECT_NAME --mae --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}.log &
    wait
done