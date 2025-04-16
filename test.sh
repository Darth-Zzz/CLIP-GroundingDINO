# GPU_NUM=1
CFG=./config/cfg_odvg.py
DATASETS=./config/refcoco.json
OUTPUT_DIR=/data_hdd/zhouzizheng/test
# NNODES=${NNODES:-1}
# NODE_RANK=${NODE_RANK:-0}
# PORT=${PORT:-29500}
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

python main.py \
        --output_dir ${OUTPUT_DIR} \
        --eval \
        -c ${CFG} \
        --datasets ${DATASETS}  \
        --pretrain_model_path /data_hdd/zhouzizheng/checkpoints/checkpoint_best_regular.pth \
        --options text_encoder_type=./bert-base-uncased
