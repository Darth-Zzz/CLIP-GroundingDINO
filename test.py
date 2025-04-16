# import requests
# import torch
# import clip
# from PIL import Image
# from torchvision import transforms

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# #  image = preprocess(Image.open(requests.get(url, stream=True).raw)).unsqueeze(0).to(device)
# image = preprocess(Image.open("/data_hdd/zhouzizheng/data/DetectEverything/train/1_army_flag.jpg")).unsqueeze(0).to(device)
# # print(image.shape)
# text = clip.tokenize(["ExcelScreen . Dollar bill . EUR bill . middle finger . pdd logo . red popup . oppo . vivo . ZTE . Redmi . xiaomi letter . TPLink . Letv . Gionee . TCL . DASH . BTC . ETH . LTC . DOGE . army flag . FaLunGong . Cross . RMB coin . LG . PDB . Haier ."]).to(device)
# print(text.shape)

# with torch.no_grad():
#     print(image.shape)
#     image_features = model.encode_image(image)
#     # print(image_features.shape)
#     text_features = model.encode_text(text)
#     # print(text_features.shape)
    
#     # logits_per_image, logits_per_text = model(image, text)
#     # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# label_list = ['3kParty', 'Fatima', 'Nazi', 'SMZDM', 'Promotion elements', 'Computer Monitor', 'Button', 'apple', 'Mobile phone', 'HP', 'Computer Mainframe', 'Thunoerobot', 'Red Arrow', 'close', 'PHILIPS', 'ThinkPad', 'iphone camera', 'Changhong', 'Hexagram', 'XingYue', 'SevenCandlesticks', 'national flag', 'Pentagram', 'ABC', 'Ad words', 'WeChat Pay', 'mosaic', 'ICBC', 'BOCOM', 'CMBC', 'national emblem', 'CMCC', 'Tsinghua university', 'Peking university', 'BOC', 'Party', 'Police badge', 'army emblem', 'League emblem', 'CCB', 'watermark text', 'PSBC', 'watermark logo', 'lenovo', 'huawei Logo', 'huawei', 'CT', 'CCTV', 'xiaomi', 'Douyin', 'kuaishou logo', 'Red Circle', 'Popups', 'candle', 'Indians', 'Rainbow flag', 'BSW flag', 'Guanyin', 'Buddha', 'WanSign', 'Razer', 'Dell', 'MSI', 'ROG', 'Alienware', 'samsung', 'ASUS', 'Acer', 'honor', 'meizu', 'EUR coin', "'ww", 'American flag', 'Russia flag', 'Popups download', 'Hop', 'ChatScreen', 'iphone key', 'Play', 'CUCC', 'ExcelScreen', 'Dollar bill', 'EUR bill', 'middle finger', 'pdd logo', 'red popup', 'oppo', 'vivo', 'ZTE', 'Redmi', 'xiaomi letter', 'TPLink', 'Letv', 'Gionee', 'TCL', 'DASH', 'BTC', 'ETH', 'LTC', 'DOGE', 'army flag', 'FaLunGong', 'Cross', 'RMB coin', 'LG', 'PDB', 'Haier', 'RMB bill', 'CircleFriends', 'Induce click calling', 'taobao logo', 'Olympic', 'jd logo', 'candle 64', 'snake', 'LionOfJudah', 'HKdollar bill', 'HKdollar coin', 'Dollar coin', 'message window', 'Farawaha', 'huawei share', 'Microsoft', 'click', 'SHIB', 'RedLionDay', 'RedCrescent', 'Yellow umbrella', 'League flag', 'Hunan tv', 'Cisco', 'Palestinian flag', 'Santa', 'Zombie', 'False msg tip', 'Honor of Kings', 'Tencent', 'Hisense', 'cigarette', 'iphone cameta', 'XRP', 'hamas flag', 'Colorful', 'Americal flag', 'Aryan flag', 'Headquarters flag', 'Jihad movement flag', 'H3C', 'Nazi salute', 'drug syringe', 'realme']
# print(len(label_list))

# from transformers import AutoTokenizer, CLIPTextModel
# import torch.nn as nn


# model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
# tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# special_tokens = tokenizer.convert_tokens_to_ids(["<|startoftext|>", "<|endoftext|>", ".", "?",])
# print(special_tokens)
# input_ids = tokenizer(".", return_tensors="pt")["input_ids"]
# print(input_ids)
# dot = tokenizer.convert_ids_to_tokens([13])
# dot2 = tokenizer.convert_ids_to_tokens([269])
# print(dot, dot2)
# inputs = tokenizer(["a photo of a cat woqbd poqn pojdposxjk owp q q nopsqq", "a photo of a dog"], padding=True, return_tensors="pt")
# print(inputs)
# tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
# print(tokens)
# outputs = model(**inputs)
# print(outputs)
# last_hidden_state = outputs.last_hidden_state
# print(last_hidden_state.shape)
# pooled_output = outputs.pooler_output 
# print(pooled_output.shape)

# specical_tokens = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
# print(specical_tokens)

# from PIL import Image
# import requests
# from transformers import AutoProcessor, CLIPVisionModel

# model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# image.save("test.jpg")
# image = image.convert("RGB")
# image.save("test2.jpg")

# inputs = processor(images=image, return_tensors="pt")

# outputs = model(**inputs)
# print(inputs)
# print(outputs)
# last_hidden_state = outputs.last_hidden_state
# print(last_hidden_state.shape)
# pooled_output = outputs.pooler_output  # pooled CLS states

# from pycocotools.coco import COCO
# coco = COCO("/data_hdd/zhouzizheng/data/Objects365-2020/train/zhiyuan_objv2_train.json")
# catIds = coco.getCatIds()
# images = {catId: coco.loadImgs(coco.getImgIds(catIds=catId)) for catId in catIds}
# for catId in catIds:
#     print(catId, len(images[catId]))

import json
# import torch
anno = "/data_hdd/zhouzizheng/data/GQA/final_mixed_train_no_coco_odvg.jsonl"
# anno = "/data_hdd/zhouzizheng/data/refCOCO/refcoco_odvg.jsonl"
with open(anno, 'r')as f:
    metas = [json.loads(line) for line in f]

for meta in metas:
    instances = meta['grounding']['regions']
    boxes = [obj["bbox"] for obj in instances]
    for box in boxes:
        if len(box) != 4:
            print(meta['filename'], len(box), boxes)
            break
    # boxes = torch.as_tensor(boxes, dtype=torch.float32)
    # print(boxes.shape)
    # boxes = boxes.reshape(-1, 4)
    

# print(sum([1 for meta in metas if len(meta['detection']['instances']) == 0]))

# from pycocotools.coco import COCO
# import os
# coco = COCO("/data_hdd/zhouzizheng/data/Objects365-2020/images/zhiyuan_objv2_train.json")
# import shutil
# root = "/data_hdd/zhouzizheng/data/Objects365-2020/train"
# images = coco.loadImgs(coco.getImgIds())
# for image in images:
#     print(image['file_name'])
#     file_name, dir_name = os.path.basename(image['file_name']), os.path.dirname(image['file_name'])
#     os.makedirs(dir_name, exist_ok=True)
#     old_dir_name = os.path.dirname(dir_name)
#     shutil.move(os.path.join(root, old_dir_name, file_name), os.path.join(root, dir_name))

    

