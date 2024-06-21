#!/bin/bash
# # HAVO
# PROJECT_DIR=/shared/ssd_16T/yl535/project/python/
PROJECT_DIR=/data/home/luoy/project/python/
TASK=cls # md | tds | cls
MODEL_TYPE=( resnet ) # efficientnet | vit | resnet | swin | vgg | resnext | wideresnet | efficientnetv1 | convnext
LOSS_TYPE='bce' # mse | cos | kld | mae | gaussnll | bce 
LR=5e-4 # ( 1e-5 5e-5 1e-4 5e-4 ) # 5e-4 best for oct bscans # 5e-5 for rnflt with efficientnet | 2e-5 for rnflt with resnet | 1e-4 for ilm
NUM_EPOCH=10
# BATCH_SIZE=16 # 16 is best
BATCH_SIZE=6 # 20 # ( 18 24 16 ) # best 6 for rnflt, 20 for resnet
STRETCH_RATIO=5 #( 0.5 1 2 5 10 26 ) # best 5
MODALITY_TYPE='oct_bscans_3d' # 'rnflt' | 'bscans' | 'oct_bscans' | 'slo_fundus' | 'oct_bscans_3d'
ATTRIBUTE_TYPE=( race gender hispanic ) # race|gender|hispanic|maritalstatus|language

if [ ${MODALITY_TYPE} = 'rnflt' ]; then
	LR=5e-4 # 5e-5 for rnflt with efficientnet
	BATCH_SIZE=6 # 6 for rnflt with efficientnet | 20 for rnflt with resnet
elif [ ${MODALITY_TYPE} = 'oct_bscans' ]; then
	LR=5e-4 # 5e-5 for rnflt with efficientnet
	BATCH_SIZE=6 # 6 for rnflt with efficientnet | 20 for rnflt with resnet
elif [ ${MODALITY_TYPE} = 'oct_bscans_3d' ]; then
	LR=5e-4 # 5e-5 for rnflt with efficientnet
	BATCH_SIZE=6 # 6 for rnflt with efficientnet | 20 for rnflt with resnet
elif [ ${MODALITY_TYPE} = 'slo_fundus' ]; then
	LR=5e-5 # 5e-5 for rnflt with efficientnet
	BATCH_SIZE=6
else
	LR=5e-5 # 5e-5 for rnflt with efficientnet
	BATCH_SIZE=6
fi

SPLIT_FILE=split.csv # ( split1064.csv split5374.csv split4934.csv )
IMBALANCE_BETA=0.9999
IMBALANCE_BETA=-1
# SUBSET_NAME=( val_3k_seed1126 val_3k_seed1458 val_3k_seed1980 val_3k_seed2793 val_3k_seed3839 val_3k_seed4117 val_3k_seed5767 val_3k_seed7477 val_3k_seed9547 val_3k_seed9667 )
SUBSET_NAME=test
NEED_BALANCE=True
DATASET_PROPORTION=1. # ( 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 )
# LR=( 1e-5 2e-5 1e-4 )
# ${#PROGRESSION_TYPE[@]}  ${#SUBSET_NAMES[@]}
for (( j=0; j<${#ATTRIBUTE_TYPE[@]}; j++ ));
do
PERF_FILE=${MODEL_TYPE}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE[$j]}.csv
for (( i=0; i<3; i++ ));
do
# python train_glaucoma_fair_allattr.py \
# 		--data_dir ${PROJECT_DIR}/datasets/harvard/glaucoma_ophthalmology_journal_9648/ \
# 		--result_dir ./results_ophjournal/glaucoma_${MODALITY_TYPE}/fullysup_${MODEL_TYPE[$j]}_${MODALITY_TYPE}_Task${TASK}_lr${LR}_bz${BATCH_SIZE}_beta${IMBALANCE_BETA}_${SUBSET_NAME} \
# 		--model_type ${MODEL_TYPE[$j]} \
# 		--image_size 200 \
# 		--loss_type ${LOSS_TYPE} \
# 		--lr ${LR} --weight-decay 0. --momentum 0.1 \
# 		--batch-size ${BATCH_SIZE} \
# 		--task ${TASK} \
# 		--epochs ${NUM_EPOCH} \
# 		--modality_types ${MODALITY_TYPE} \
# 		--split_seed 5 \
# 		--imbalance_beta ${IMBALANCE_BETA} \
# 		--perf_file ${PERF_FILE} \
# 		--attribute_type ${ATTRIBUTE_TYPE} \
# 		--subset_name ${SUBSET_NAME} 
# 		# --seed 13 \
python train_glaucoma_fair_allattr_withsplit.py \
		--data_dir /data/home/shim/pyspace/fairness/dataset/ \
		--result_dir ./results_year2123_3d/glaucoma_${MODALITY_TYPE}_${ATTRIBUTE_TYPE[$j]}_balance/fullysup_${MODEL_TYPE}_${MODALITY_TYPE}_Task${TASK}_lr${LR}_bz${BATCH_SIZE}_beta${IMBALANCE_BETA}_${SUBSET_NAME} \
		--model_type ${MODEL_TYPE} \
		--image_size 200 \
		--loss_type ${LOSS_TYPE} \
		--lr ${LR} --weight-decay 0. --momentum 0.1 \
		--batch-size ${BATCH_SIZE} \
		--task ${TASK} \
		--epochs ${NUM_EPOCH} \
		--modality_types ${MODALITY_TYPE} \
		--split_seed 5 \
		--imbalance_beta ${IMBALANCE_BETA} \
		--perf_file ${PERF_FILE} \
		--attribute_type ${ATTRIBUTE_TYPE[$j]} \
		--subset_name ${SUBSET_NAME} \
		--need_balance ${NEED_BALANCE} \
		--dataset_proportion ${DATASET_PROPORTION} \
		--split_file ${SPLIT_FILE}
		# --data_dir ${PROJECT_DIR}/datasets/harvard/fairness_for_allyears/ \
		# --data_dir ${PROJECT_DIR}/datasets/harvard/glaucoma_ophthalmology_journal_9648/ \
		# --data_dir ${PROJECT_DIR}/datasets/harvard/glaucoma_lancet_journal_5612_3300/ \
done
done
