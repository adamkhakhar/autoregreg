GPU_IND=7
NUM_SAMPLES=200000
BATCH_SIZE=1000
NUM_WORKERS=3
LOG_EVERY=10

mkdir -p ../../results/result_logging/
for LEARNING_RATE in .1 .01 .005 .001 .0005 .0001 .00005 .00001 .000005 .000001 
do
    echo LEARNING_RATE $LEARNING_RATE
    name=mnist_mae_sin_s_log_s
    echo RUNNING $name
    python3 ../mnist_experiments/mnist_mae_sin_s_log_s.py ${name} --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}.log &
    wait
done