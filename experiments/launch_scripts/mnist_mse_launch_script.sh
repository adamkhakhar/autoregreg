GPU_IND=4
NUM_SAMPLES=1000000
BATCH_SIZE=1000
NUM_WORKERS=2
LOG_EVERY=100

mkdir -p ../../results/result_logging/
for iteration in {1..10..1}
do
    echo iteration $iteration
    learning_rate=.0001
    name=mnist_mse_sin_s_log_s
    echo RUNNING $name
    python3 ../mnist_experiments/mnist_mse_sin_s_log_s.py ${name} --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $learning_rate --seed $iteration >> ../../results/result_logging/${name}.log &

    learning_rate=.005

    name=mnist_mse_sin_s_log_l
    echo RUNNING $name
    python3 ../mnist_experiments/mnist_mse_sin_s_log_l.py ${name} --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $learning_rate --seed $iteration  >> ../../results/result_logging/${name}.log &

    name=mnist_mse_log_s_sin_l
    echo RUNNING $name
    python3 ../mnist_experiments/mnist_mse_log_s_sin_l.py ${name} --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $learning_rate --seed $iteration  >> ../../results/result_logging/${name}.log &

    wait
done