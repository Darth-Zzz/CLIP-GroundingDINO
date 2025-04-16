CFG=./config/cfg_odvg.py
DATASETS=./config/test.json
OUTPUT_DIR=/data_hdd/zhouzizheng/test
# NNODES=${NNODES:-1}
# NODE_RANK=${NODE_RANK:-0}
# PORT=${PORT:-29500}
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Change ``pretrain_model_path`` to use a different pretrain. 
# (e.g. GroundingDINO pretrain, DINO pretrain, Swin Transformer pretrain.)
# If you don't want to use any pretrained model, just ignore this parameter.

python main.py \
        --output_dir ${OUTPUT_DIR} \
        -c ${CFG} \
        --datasets ${DATASETS}  \
        --options text_encoder_type=bert-base-uncased \
        # --pretrain_model_path pretrained/groundingdino_swint_ogc.pth 
