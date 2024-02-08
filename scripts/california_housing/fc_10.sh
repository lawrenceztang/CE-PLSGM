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
run2() {
echo "Running with seed=$2"
        python src/train.py \
	--optimizer_name ce_plsgm \
        --n_global_iters 500 \
        --eps -1 \
        --model_name $model_name \
        --exp_name $exp_name \
        --delta .00000001 \
        --seed $2
}


echo "Running with eps=-1, seed=$1"
mkdir -p out/$exp_name/eps$eps/$dataset_name/seed$1/$model_name
run2 -1 $1 > out/$exp_name/eps$eps/$dataset_name/seed$1/$model_name/ce_plsgm.out


# Sub-routine for DP-GD & Diff2
run() {
    for c in 1 3 10 30 100
    do
echo "Running with c=$c, seed=$2"
        python src/train.py \
	--optimizer_name ce_plsgm \
        --n_global_iters 500 \
        --eps $eps \
        --model_name $model_name \
        --exp_name $exp_name \
        --delta .00000001 \
        --c $c \
        --seed $2
    done
}


for eps in -1 .6 1.0 1.8
do
    echo "Running with eps=$eps, seed=$1"
    mkdir -p out/$exp_name/eps$eps/$dataset_name/seed$1/$model_name
    run $eps $1 > out/$exp_name/eps$eps/$dataset_name/seed$1/$model_name/ce_plsgm.out
done

