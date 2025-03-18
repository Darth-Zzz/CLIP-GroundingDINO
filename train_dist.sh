GPU_NUM=1
CFG=./config/cfg_odvg.py
DATASETS=./config/test.json
OUTPUT_DIR=/data_hdd/zhouzizheng/checkpoints
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Change ``pretrain_model_path`` to use a different pretrain. 
# (e.g. GroundingDINO pretrain, DINO pretrain, Swin Transformer pretrain.)
# If you don't want to use any pretrained model, just ignore this parameter.

python -m torch.distributed.launch  --nproc_per_node=${GPU_NUM} main.py \
        --output_dir ${OUTPUT_DIR} \
        -c ${CFG} \
        --datasets ${DATASETS}  \
        --pretrain_model_path pretrained/groundingdino_swint_ogc.pth \
        --options text_encoder_type=bert-base-uncased
