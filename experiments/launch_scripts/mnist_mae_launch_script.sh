GPU_IND=0
NUM_SAMPLES=10000000
BATCH_SIZE=1000
NUM_WORKERS=4
LOG_EVERY=100
learning_rate=.01

mkdir -p ../../results/result_logging/
echo learning_rate $learning_rate
name=mnist_mae_log_s_sin_l
echo RUNNING $name
python3 ../mnist_mae.py ${name} --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $learning_rate --seed 1 >> ../../results/result_logging/${name}.log &
