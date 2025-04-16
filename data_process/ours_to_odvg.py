import xml.etree.ElementTree as ET
import os
import json
from pycocotools.coco import COCO
import pickle


train_ratio = 0.8

def corresponding_img(xml_file):
    img_suffix = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    for suffix in img_suffix:
        if os.path.exists(xml_file.replace('.xml', suffix)):
            return xml_file.replace('.xml', suffix)
    return None



def ours2coco(input_root, output_root):
    image_id = 0
    box_id = 0
    category_id = {}
    category_index = 1
    os.makedirs(f'{output_root}/annotations', exist_ok=True)
    train_image_list = []
    val_image_list = []
    images = {
        'train': [],
        'val': []
    }
    annotations = {
        'train': [],
        'val': []
    }
    categories = []

    for dir_name in os.listdir(input_root):
        dir_path = os.path.join(input_root, dir_name)
        all_xml_files = [file_name for file_name in os.listdir(dir_path) if file_name.endswith('.xml')]
        train_xml_files = all_xml_files[:int(len(all_xml_files) * train_ratio)]
        val_xml_files = all_xml_files[int(len(all_xml_files) * train_ratio):]
        train_image_list.extend([corresponding_img(os.path.join(dir_path, file_name)) for file_name in train_xml_files])
        val_image_list.extend([corresponding_img(os.path.join(dir_path, file_name)) for file_name in val_xml_files])
        for phase in ["train", "val"]:
            xml_files = train_xml_files if phase == "train" else val_xml_files
            for file_name in xml_files:
                file_path = os.path.join(dir_path, file_name)
                tree = ET.parse(file_path)
                root = tree.getroot()
                size = root.find('size')
                width = int(size.find('width').text)
                height = int(size.find('height').text)
                image = {
                    'id': image_id,
                    'file_name': os.path.basename(corresponding_img(file_path)),
                    'width': width,
                    'height': height
                }
                images[phase].append(image)
                for obj in root.iter('object'):
                    name = obj.find('name').text.replace('_', ' ')
                    bndbox = obj.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    w = xmax - xmin
                    h = ymax - ymin
                    area = w * h
                    if name not in category_id:
                        category_id[name] = category_index
                        category = {
                            'id': category_index,
                            'name': name,
                            'supercategory': name
                        }
                        categories.append(category)
                        category_index += 1

                    annotation = {
                        'id': box_id,
                        'image_id': image_id,
                        'category_id': category_id[name],
                        'bbox': [xmin, ymin, w, h],
                        'area': area,
                        'iscrowd': 0
                    }
                    box_id += 1
                    annotations[phase].append(annotation)
                image_id += 1

    for phase in ["train", "val"]:
        data = {
            'info': {},
            'licenses': [],
            'images': images[phase],
            'annotations': annotations[phase],
            'categories': categories
        }
        json.dump(data, open(f'{output_root}/annotations/coco_{phase}.json', 'w'), indent=4)
    # print(train_image_list)
    # print(val_image_list)
    return train_image_list, val_image_list


def copy_images(image_list, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for image in image_list:
        os.system(f'cp "{image}" "{dst_dir}"')
    

def coco2odvg(input_file, output_file, label_map_file):
    coco = COCO(input_file)
    id_to_category = {category['id']: category['name'] for category in coco.loadCats(coco.getCatIds())}
    new_id_to_category = sorted(id_to_category.items(), key=lambda x: x[0])
    label_map = {}
    label_map["0"] = "REF OBJECT"
    label_map.update({str(new_id_to_category.index(item)+1): item[1] for item in new_id_to_category})
    
    json.dump(label_map, open(label_map_file, 'w'), indent=4)

    metas = []
    for image_id, image in coco.imgs.items():
        file_name = image['file_name']
        width = image['width']
        height = image['height']
        instances = []
        for ann_id in coco.getAnnIds(imgIds=image_id):
            ann = coco.anns[ann_id]
            label = ann['category_id']
            category = id_to_category[label]
            new_label = str(new_id_to_category.index((label, category)))
            xmin, ymin, w, h = ann['bbox']
            xmax = xmin + w
            ymax = ymin + h

            instances.append({
                'label': new_label,
                'category': category,
                'bbox': [xmin, ymin, xmax, ymax]
            })
        meta = {
            "filename": file_name,
            "height": height,
            "width": width,
            "detection":{
                "instances": instances
            }
        }
        metas.append(meta)
    with open(output_file, 'w') as f:
        for meta in metas:
            f.write(json.dumps(meta) + '\n')
    

def refcoco2odvg(ref_file, coco_ann_file, output_file):
    """
    Target format:

    {
        "filename": "image.jpg",
        "height": 693,
        "width": 1024,
        "grounding": { 
            "regions": [
                {
                    "bbox": [20,215,985,665],   # [x1,y1,x2,y2]
                    "phrase": "a paper cover that reads we heart our customers",
                    "category_id": 1,               // USED FOR MATCHING CORRESPONDING SUPPORT IMAGE
                },
                { 
                    "bbox": [19,19,982,671],
                    "phrase": "a wire hanger"
                    "category_id": 1,               // USED FOR MATCHING CORRESPONDING SUPPORT IMAGE
                }
            ]
            }
    }
    """
    refs = pickle.load(open(ref_file, 'rb'))
    coco = COCO(coco_ann_file)
    id_to_category = {category['id']: category['name'] for category in coco.loadCats(coco.getCatIds())}
    new_id_to_category = sorted(id_to_category.items(), key=lambda x: x[0])

    metas = []
    img2refs, ann2ref = {}, {}
    for ref in refs:
        ann_id = ref['ann_id']
        image_id = ref['image_id']
        img2refs[image_id] = img2refs.get(image_id, []) + [ref]
        ann2ref[ann_id] = ann2ref.get(ann_id, []) + [ref]
        
    for image_id in img2refs:
        image = coco.imgs[image_id]
        file_name = image['file_name']
        width = image['width']
        height = image['height']
        regions = []
        for ref in img2refs[image_id]:
            ann_id = ref['ann_id']
            ann = coco.anns[ann_id]
            category_id = ref['category_id']
            
            xmin, ymin, w, h = ann['bbox']
            xmax = xmin + w
            ymax = ymin + h
            for sentence in ref['sentences']:
                phrase = sentence['raw']
                regions.append({
                    'bbox': [xmin, ymin, xmax, ymax],
                    'category_id': category_id,
                    'phrase': phrase
                })
        meta = {
            "filename": file_name,
            "height": height,
            "width": width,
            "grounding":{
                "regions": regions
            }
        }
        metas.append(meta)
    with open(output_file, 'w') as f:
        for meta in metas:
            f.write(json.dumps(meta) + '\n')
    
        
    
if __name__ == '__main__':
    # Ours
    train_image_list, val_image_list = ours2coco('/data_hdd/zhouzizheng/data/万物识别数据_20241213/data_cut', '/data_hdd/zhouzizheng/data/DetectEverything')
    copy_images(train_image_list, '/data_hdd/zhouzizheng/data/DetectEverything/train')
    copy_images(val_image_list, '/data_hdd/zhouzizheng/data/DetectEverything/val')
    coco2odvg('/data_hdd/zhouzizheng/data/DetectEverything/annotations/coco_train.json', '/data_hdd/zhouzizheng/data/DetectEverything/annotations/odvg_train.jsonl', '/data_hdd/zhouzizheng/data/DetectEverything/annotations/label_map_train.json')

    # RefCOCO
    refcoco2odvg("/data_hdd/zhouzizheng/data/refCOCO/refcoco/refs(unc).p", "/data_hdd/zhouzizheng/data/refCOCO/refcoco/instances.json", "/data_hdd/zhouzizheng/data/refCOCO/refcoco_odvg.jsonl")
    refcoco2odvg("/data_hdd/zhouzizheng/data/refCOCO/refcoco+/refs(unc).p", "/data_hdd/zhouzizheng/data/refCOCO/refcoco+/instances.json", "/data_hdd/zhouzizheng/data/refCOCO/refcoco+_odvg.jsonl")
    refcoco2odvg("/data_hdd/zhouzizheng/data/refCOCO/refcocog/refs(umd).p", "/data_hdd/zhouzizheng/data/refCOCO/refcocog/instances.json", "/data_hdd/zhouzizheng/data/refCOCO/refcocog_odvg.jsonl")

    # O365
    coco2odvg("/data_hdd/zhouzizheng/data/Objects365-2020/test/images/zhiyuan_objv2_test.json", "/data_hdd/zhouzizheng/data/Objects365-2020/test/zhiyuan_objv2_test_odvg.jsonl", "/data_hdd/zhouzizheng/data/Objects365-2020/test/zhiyuan_objv2_test_label_map.json")
    coco2odvg("/data_hdd/zhouzizheng/data/Objects365-2020/val/zhiyuan_objv2_val.json", "/data_hdd/zhouzizheng/data/Objects365-2020/val/zhiyuan_objv2_val_odvg.jsonl", "/data_hdd/zhouzizheng/data/Objects365-2020/val/zhiyuan_objv2_val_label_map.json")
    coco2odvg("/data_hdd/zhouzizheng/data/Objects365-2020/train/zhiyuan_objv2_train.json", "/data_hdd/zhouzizheng/data/Objects365-2020/train/zhiyuan_objv2_train_odvg.jsonl", "/data_hdd/zhouzizheng/data/Objects365-2020/train/zhiyuan_objv2_train_label_map.json")