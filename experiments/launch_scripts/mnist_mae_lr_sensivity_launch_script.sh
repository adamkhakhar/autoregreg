GPU_IND=7
NUM_SAMPLES=10000000
BATCH_SIZE=1000
NUM_WORKERS=3
LOG_EVERY=100

mkdir -p ../../results/result_logging/
for learning_rate in .000001 .00001 .0001 .0005 .001 .005 .01 .1 1
do
    echo learning_rate $learning_rate
    for target_scale in .001 10 1000000000
    do
        echo target_scale $target_scale
        name=mnist_mae_lr_sensitivity
        echo RUNNING $name
        python3 ../mnist_experiments/mnist_mae_lr_sensitivity.py ${name} --scale_targets $target_scale --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --gpu_ind $GPU_IND --log_every $LOG_EVERY --learning_rate $learning_rate --seed 1 >> ../../results/result_logging/${name}.log &
    done
    wait
done