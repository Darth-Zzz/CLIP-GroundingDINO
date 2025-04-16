import json
src = "/data_hdd/zhouzizheng/data/Objects365-2020/val/zhiyuan_objv2_val.json"
dst = "/data_hdd/zhouzizheng/data/Objects365-2020/val/zhiyuan_objv2_val_prettified.json"
with open(src, 'r', encoding='utf-8') as f:
    json_string = f.read()

data = json.loads(json_string)
formatted_json = json.dumps(data, indent=4, ensure_ascii=False)

with open(dst, 'w', encoding='utf-8') as f:
    f.write(formatted_json)