# Basic settings
seed=0

exp_name=20230123
weight_decay=0.0
n_global_iters=500
delta=.00000001
dataset_name=california_housing
n_workers=10
model_name=fc_10
n_local_iters=1
optimizer_name=ce_plsgm
save_intvl=20
eta=None

eval "$(conda shell.bash hook)"
conda activate ce-plsgd

# Sub-routine for DP-GD & Diff2
run() {
    for c in 1 3 10 30 100
    do
	python src/train.py \
	       --eta $eta \
	       --n_workers $n_workers \
	       --seed $seed \
	       --dataset_name $dataset_name \
	       --model_name $model_name \
	       --optimizer_name $optimizer_name \
	       --exp_name $exp_name \
	       --n_global_iters $n_global_iters \
	       --n_local_iters $n_local_iters \
	       --weight_decay $weight_decay \
	       --save_intvl $save_intvl \
	       --eps $eps \
	       --delta $delta \
	       --c $c
    done
}


for eps in .6 .8 1
do
    mkdir -p out/$exp_name/eps$eps/$dataset_name/$model_name
    run $c2 $tau $eps > out/$exp_name/eps$eps/$dataset_name/$model_name.out &
done


# Sub-routine for noiseless GD

eps=None
c=None
c2=None
tau=0.0

run2() {
    python src/train.py \
	   --eta $eta \
	   --n_workers $n_workers \
	   --seed $seed \
	   --dataset_name $dataset_name \
	   --model_name $model_name \
	   --optimizer_name $optimizer_name \
	   --exp_name $exp_name \
	   --gpu_id $gpu_id \
	   --n_global_iters $n_global_iters \
	   --n_local_iters $n_local_iters \
	   --weight_decay $weight_decay \
	   --save_intvl $save_intvl \
	   --eps $eps \
	   --c $c \
	   --c2 $c2 \
	   --tau $tau
}

mkdir -p out/$exp_name/eps$eps/$dataset_name/$model_name
eta=None
run2 > out/$exp_name/eps$eps/$dataset_name/$model_name/eta_{$eta}_c_{$c}_c2_{$c2}_tau_{$tau}.out

