GPU_IND=5
NUM_SAMPLES=1000000
BATCH_SIZE=512
LOG_EVERY=100
INPUT_SIZE=400

mkdir -p ../../results/result_logging/
for LEARNING_RATE in .1 .01 .001 .0001 .00001 .000001 
do
    echo LEARNING_RATE $LEARNING_RATE
    name=rat_char_1000_mse
    echo RUNNING $name
    python3 ../amazon_review_experiments/amzn_mse_mae.py ${name} mse --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}.log

    name=rat_char_1000_mae
    echo RUNNING $name
    python3 ../amazon_review_experiments/amzn_mse_mae.py ${name} mae --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $LEARNING_RATE --seed 1 >> ../../results/result_logging/${name}.log 
done