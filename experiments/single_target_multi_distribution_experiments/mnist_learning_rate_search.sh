GPU_IND=3
NUM_SAMPLES=10000000
BATCH_SIZE=1000
NUM_WORKERS=6
LOG_EVERY=100
BASE=100
EXP_MIN=-3
EXP_MAX=4

mkdir -p ../../results/result_logging/
echo SAME SCALE
for LEARNING_RATE in 10 1 .5 .1 .05 .01 .005 .001 .0005 .0001 .00005 .00001 .000005 .000001 
do
    echo LEARNING_RATE $LEARNING_RATE
    name=mnist_arr_sin_s_log_s
    echo RUNNING $name
    python3 mnist_arr_sin_log.py ${name} --wandb --num_samples_error_track $BATCH_SIZE --base $BASE --exp_min $EXP_MIN --exp_max $EXP_MAX --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}.log

    echo LEARNING_RATE $LEARNING_RATE
    name=mnist_mse_sin_s_log_s
    echo RUNNING $name
    python3 mnist_mse_mae_sin_log.py ${name} --wandb --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}.log

    echo LEARNING_RATE $LEARNING_RATE
    name=mnist_mae_sin_s_log_s
    echo RUNNING $name
    python3 mnist_mse_mae_sin_log.py ${name} --wandb --mae --wandb_project mnist_mae_same_scale --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}.log
done

echo DIFFERENT_SCALE
for LEARNING_RATE in 10 1 .5 .1 .05 .01 .005 .001 .0005 .0001 .00005 .00001 .000005 .000001 
do
    echo LEARNING_RATE $LEARNING_RATE
    name=mnist_arr_sin_l_log_s
    echo RUNNING $name
    python3 mnist_arr_sin_log.py ${name} --sin l --wandb --num_samples_error_track $BATCH_SIZE --base $BASE --exp_min $EXP_MIN --exp_max $EXP_MAX --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}.log

    echo LEARNING_RATE $LEARNING_RATE
    name=mnist_mse_sin_l_log_s
    echo RUNNING $name
    python3 mnist_mse_mae_sin_log.py ${name} --sin l --wandb --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}.log

    echo LEARNING_RATE $LEARNING_RATE
    name=mnist_mae_sin_l_log_s
    echo RUNNING $name
    python3 mnist_mse_mae_sin_log.py ${name} --sin l --wandb --wandb_project mnist_mae_dif_scale --mae --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}.log
done