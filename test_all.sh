
CFG=./config/cfg_odvg.py
DATASETS=./config/test.json
OUTPUT_DIR=/data_hdd/zhouzizheng/results
CUDA_VISIBLE_DEVICES=3
for ckpt in /data_hdd/zhouzizheng/checkpoints/checkpoint0*.pth; do
    python main.py \
            --output_dir ${OUTPUT_DIR} \
            --eval \
            -c ${CFG} \
            --datasets ${DATASETS}  \
            --pretrain_model_path ${ckpt} \
            --options text_encoder_type=./bert-base-uncased \
            > ${ckpt}.log
    echo "Done with ${ckpt}"
done

