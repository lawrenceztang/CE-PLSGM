# Basic settings
exp_name=20230123
n_global_iters=500
delta=.00000001
dataset_name=california_housing
n_workers=10
model_name=fc_10
optimizer_name=ce_plsgm
save_intvl=20
eta=None

eval "$(conda shell.bash hook)"
conda activate ce-plsgm


# Sub-routine for DP-GD & Diff2
run() {
    for c in 1 3 10 30 100
    do
echo "Running with c=$c"
        python src/train.py \
	--optimizer_name ce_plsgm \
        --n_global_iters 500 \
        --eps $eps \
        --delta .00000001 \
        --c $c \
        --seed $1
    done
}


for eps in .6 1.2 1.8
do
    echo "Running with eps=$eps$"
    mkdir -p out/$exp_name/eps$eps/$dataset_name
    run $eps > out/$exp_name/eps$eps/$dataset_name/$model_name/ce_plsgm.out
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
	   --seed $1 \
	   --dataset_name $dataset_name \
	   --model_name $model_name \
	   --optimizer_name diff2_gd \
	   --exp_name $exp_name \
	   --n_global_iters $n_global_iters \
	   --save_intvl $save_intvl \
	   --eps $eps \
	   --c $c \
	   --c2 $c2 \
	   --tau $tau
}


eta=None
run2 $1 > out/$exp_name/eps$eps/$dataset_name/$model_name/eta_{$eta}_c_{$c}_c2_{$c2}_tau_{$tau}.out
