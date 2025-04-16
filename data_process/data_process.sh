# flickr30k entities
python flickr30ke2odvg.py \
    --root /data_hdd/zhouzizheng/data/Flickr30kEntities/ \
    --output_file /data_hdd/zhouzizheng/data/Flickr30kEntities/flickr30k_e2odvg.jsonl
# gqa
python gqa2odvg.py /data_hdd/zhouzizheng/data/GQA/final_mixed_train_no_coco.json

# 万物识别数据_20241213, refcoco, refcoco+, refcocog, objects 365
python ours_to_odvg.py