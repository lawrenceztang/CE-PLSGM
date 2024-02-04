model_name=fc_10

for seed in 0 1 2 3 4
do
    for dataset_name in california_housing
    do
        bash scripts/${dataset_name}/${model_name}.sh $seed
    done
done

