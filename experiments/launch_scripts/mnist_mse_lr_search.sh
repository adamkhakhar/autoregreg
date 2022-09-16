GPU_IND=4
NUM_SAMPLES=100000
BATCH_SIZE=1000
NUM_WORKERS=3
LOG_EVERY=100

mkdir -p ../../results/result_logging/
for LEARNING_RATE in .01 .005 .001 .0005 .0001 .00005 .00001 .000005 .000001 
do
    echo LEARNING_RATE $LEARNING_RATE
    name=mnist_mse_sin_s_log_s
    echo RUNNING $name
    python3 ../mnist_experiments/mnist_mse_sin_s_log_s.py ${name} --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed $iteration >> ../../results/result_logging/${name}.log &

    name=mnist_mse_sin_s_log_l
    echo RUNNING $name
    python3 ../mnist_experiments/mnist_mse_sin_s_log_l.py ${name} --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed $iteration >> ../../results/result_logging/${name}.log &
    wait
done