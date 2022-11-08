GPU_IND=7
NUM_SAMPLES=100000000
BATCH_SIZE=10000
NUM_WORKERS=1
LOG_EVERY=100

mkdir -p ../../results/result_logging/
for iteration in {1..10..1}
do
    echo iteration $iteration
    learning_rate=.005
    name=mae_sin_s_log_s
    echo RUNNING $name
    python3 ../one_dim_experiments/one_dim_mae_sin_s_log_s.py ${name} --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $learning_rate --seed $iteration >> ../../results/result_logging/${name}.log &

    learning_rate=.001

    name=mae_sin_s_log_l
    echo RUNNING $name
    python3 ../one_dim_experiments/one_dim_mae_sin_s_log_l.py ${name} --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $learning_rate --seed $iteration  >> ../../results/result_logging/${name}.log &

    name=mae_log_s_sin_l
    echo RUNNING $name
    python3 ../one_dim_experiments/one_dim_mae_log_s_sin_l.py ${name} --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $learning_rate --seed $iteration  >> ../../results/result_logging/${name}.log &

    wait
done