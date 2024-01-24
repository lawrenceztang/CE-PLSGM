model_name=fc_10

for seed in 0 1 2 3 4
do
    for dataset_name in california_housing gas blog
    do
        bash scripts/${dataset_name}/${model_name}_seed$seed.sh
    done
done

