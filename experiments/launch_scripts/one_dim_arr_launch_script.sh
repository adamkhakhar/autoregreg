GPU_IND=4
BASE=100
EXP_MIN=-3
EXP_MAX=4
NUM_SAMPLES=100000000
BATCH_SIZE=10000
NUM_WORKERS=2
LOG_EVERY=100
LEARNING_RATE=.0001

mkdir -p ../results/result_logging/
for iteration in {1..10..1}
do
    echo iteration $iteration
    name=arr_sin_s_log_s
    echo RUNNING $name
    python3 ../one_dim_experiments/one_dim_arr_sin_s_log_s.py ${name} --base $BASE --exp_min $EXP_MIN --exp_max $EXP_MAX --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed $iteration >> ../results/result_logging/${name}.log &

    name=arr_sin_s_log_l
    echo RUNNING $name
    python3 ../one_dim_experiments/one_dim_arr_sin_s_log_l.py ${name} --base $BASE --exp_min $EXP_MIN --exp_max $EXP_MAX --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed $iteration  >> ../results/result_logging/${name}.log &

    name=arr_log_s_sin_l
    echo RUNNING $name
    python3 ../one_dim_experiments/one_dim_arr_log_s_sin_l.py ${name} --base $BASE --exp_min $EXP_MIN --exp_max $EXP_MAX --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed $iteration  >> ../results/result_logging/${name}.log &
    wait
done