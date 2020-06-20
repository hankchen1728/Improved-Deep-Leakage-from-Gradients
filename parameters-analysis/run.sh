# Set global parameter
DATA_DIR='../../data/CIFAR10'
EXP_DIR='../../exp'
EXP_NAME=$1
SEED=30 # 42, 97
MAX_ITERS=300
MSE_TOL=3.0

IMAGE_IDX=10
SELECT_IMAGE_INDICES=(
	0 1 2 3 4 5 6 7 8 9 \
	10 11 12 13 14 15 16 17 18 19 \
	20 21 22 23 24 25 26 27 28 29 \
	30 31 32 33 34 35 36 37 38 39 \
	40 41 42 43 44 45 46 47 48 49 \
	50 51 52 53 54 55 56 58 59 60 \
	61 62 64 65 66 67 68 69 70 71 \
	72 73 74 77 81 82 83 84 85 86 \
	89 92 93 95 100 103 104 106 107 \
	111 115 116 117 128 129 135 139 \
	148 155 165)


for i in "${SELECT_IMAGE_INDICES[@]}"
do
	echo "Analyzing image $i"
	python3 analysis.py \
		--visible-gpus $2 \
		--seed $3 \
		--data-dir $DATA_DIR \
		--exp-dir $EXP_DIR \
		--exp-name $EXP_NAME \
		--image-idx $i \
		--max-iters $MAX_ITERS \
		--mse-tol $MSE_TOL
done