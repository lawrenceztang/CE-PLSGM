# Basic settings
gpu_id=-1
seed=0

exp_name=20230123
weight_decay=0.0
n_global_iters=2000
dataset_name=california_housing
n_workers=10
model_name=fc_10
n_local_iters=1
optimizer_name=diff2_gd
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
	       --gpu_id $gpu_id \
	       --n_global_iters $n_global_iters \
	       --n_local_iters $n_local_iters \
	       --weight_decay $weight_decay \
	       --save_intvl $save_intvl \
	       --eps $3 \
	       --c $c \
	       --c2 $1 \
	       --tau $2
    done
}


for eps in 3 5
do
    mkdir -p out/$exp_name/eps$eps/$dataset_name/$model_name
    for c2 in 1 3 10 30 100
    do
	for tau in 0.003 0.01 0.03 0.1
	do
	    run $c2 $tau $eps > out/$exp_name/eps$eps/$dataset_name/$model_name/c2_${c2}_tau_${tau}.out &
	done
    done
    sleep 1
    
    c2=None
    tau=0.0
    run $c2 $tau $eps > out/$exp_name/eps$eps/$dataset_name/$model_name/c2_${c2}_tau_${tau}.out
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

