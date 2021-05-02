import os
import json
import datetime
from PIL import Image
import numpy as np
from itertools import groupby
from skimage import measure
from pycocotools import mask
from tqdm import tqdm
import glob
import shutil
import cv2 as cv
def process_obj_info(obj_info):
    obj_dict = {}
    obj_classes = obj_info["exported_objects"]
    for i in obj_classes:
        obj_dict[i['class']]= i['segmentation_class_id']
    return obj_dict

def setCOCOJibberJabber():
    INFO = {
        "description": "FAT Dataset",
        "url": "",
        "version": "0.1.0",
        "contributor": "ruolin",
        "date_created": datetime.datetime.utcnow().isoformat(" ")
    }

    LICENSES = [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ]
    CLASSES = ('__background__', '002_master_chef_can_16k', '003_cracker_box_16k', '004_sugar_box_16k', '005_tomato_soup_can_16k', '006_mustard_bottle_16k',
               '007_tuna_fish_can_16k', '008_pudding_box_16k', '009_gelatin_box_16k', '010_potted_meat_can_16k', '011_banana_16k', '019_pitcher_base_16k',
               '021_bleach_cleanser_16k', '024_bowl_16k', '025_mug_16k', '035_power_drill_16k', '036_wood_block_16k', '037_scissors_16k', '040_large_marker_16k',
               '051_large_clamp_16k', '052_extra_large_clamp_16k', '061_foam_brick_16k')
    CATEGORIES = [dict(zip(['id','name','supercategory'], [ix,c,'FAT'])) for ix, c in enumerate(CLASSES)]


    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    return coco_output

def create_image_info(image_id, image_name, image_size, date_captured=datetime.datetime.utcnow().isoformat(' '), license_id=1, coco_url="", flickr_url=""):
    
    image_info = {
        "id": image_id,
        "file_name": image_name,
        "width": image_size[0],
        "height": image_size[1],
        "date_captured": date_captured,
        "license": license_id,
        "coco_url": coco_url,
        "flickr_url": flickr_url
    }

    return image_info

def generate_submasks(sem_mask_path, inst_mask_path):
    sem_mask = np.array(Image.open(sem_mask_path))
    inst_mask = np.array(Image.open(inst_mask_path))
    cat_list = []
    inst_masks = []
    crowd_list = []
    
    id_list = np.unique(inst_mask)
    for inst_id in id_list:
        if inst_id == 0 or inst_id == 255: # unless we have more than 255 category/instance in an images
            continue
        tempMask = np.zeros(inst_mask.shape)
        tempMask.fill(inst_id)
        c = (tempMask == inst_mask)
        instance = inst_mask * c
        cat_id = np.max(sem_mask * c)
        cat_list.append(cat_id)
        inst_masks.append(instance)
        crowd_list.append(False) # for some other dataset, there may be some strategies for crowd list
    return cat_list, inst_masks, crowd_list

def binary_mask_to_rle(binary_mask):
    rle = {'counts':[], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0) 
        counts.append(len(list(elements)))
    return rle

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(binary_mask, tolerance=0):
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation 
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons

def create_annotation_info(annotation_id, image_id, category_info, binary_mask, tolerance=2, bounding_box=None):
    binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    area = mask.area(binary_mask_encoded)
    if area < 1:
        return None

    if bounding_box is None:
        bounding_box = mask.toBbox(binary_mask_encoded)
    
    if category_info==1:
        is_crowd = 1
        segmentation = binary_mask_to_rle(binary_mask)
    else:
        is_crowd = 0
        segmentation = binary_mask_to_polygon(binary_mask, tolerance)
        if not segmentation:
            return None
    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": is_crowd,
        "area": area.tolist(),
        "bbox": bounding_box.tolist(),
        "segmentation": segmentation,
        "width": binary_mask.shape[1],
        "height": binary_mask.shape[0],
    }

    return annotation_info

def fix_type(o):
    if isinstance(o, np.uint8):
        return int(o)
    raise TypeError

def main(root_dir, image_dir_path, image_copy_dir, save_path, obj_info):
    coco_output = setCOCOJibberJabber()
    categories = coco_output['categories']
    """
    [{'id': 0, 'name': '__background__', 'supercategory': 'FAT'}, {'id': 1, 'name': '002_master_chef_can', 'supercategory': 'FAT'}, {'id': 2, 'name': '003_cracker_box', 'supercategory': 'FAT'}, {'id': 3, 'name': '004_sugar_box', 'supercategory': 'FAT'}, {'id': 4, 'name': '005_tomato_soup_can', 'supercategory': 'FAT'}, {'id': 5, 'name': '006_mustard_bottle', 'supercategory': 'FAT'}, {'id': 6, 'name': '007_tuna_fish_can', 'supercategory': 'FAT'}, {'id': 7, 'name': '008_pudding_box', 'supercategory': 'FAT'}, {'id': 8, 'name': '009_gelatin_box', 'supercategory': 'FAT'}, {'id': 9, 'name': '010_potted_meat_can', 'supercategory': 'FAT'}, {'id': 10, 'name': '011_banana', 'supercategory': 'FAT'}, {'id': 11, 'name': '019_pitcher_base', 'supercategory': 'FAT'}, {'id': 12, 'name': '021_bleach_cleanser', 'supercategory': 'FAT'}, {'id': 13, 'name': '024_bowl', 'supercategory': 'FAT'}, {'id': 14, 'name': '025_mug', 'supercategory': 'FAT'}, {'id': 15, 'name': '035_power_drill', 'supercategory': 'FAT'}, {'id': 16, 'name': '036_wood_block', 'supercategory': 'FAT'}, {'id': 17, 'name': '037_scissors', 'supercategory': 'FAT'}, {'id': 18, 'name': '040_large_marker', 'supercategory': 'FAT'}, {'id': 19, 'name': '051_large_clamp', 'supercategory': 'FAT'}, {'id': 20, 'name': '052_extra_large_clamp', 'supercategory': 'FAT'}, {'id': 21, 'name': '061_foam_brick', 'supercategory': 'FAT'}]
    """
    image_id = 1
    segmentation_id = 1
    
    image_dir = os.listdir(image_dir_path)
    for scene in image_dir:
        scene_path = os.path.join(root_dir,scene)
        # files = os.listdir(scene_path)
        images = glob.glob("%s/*.jpg" % (scene_path)
                          )
        for image_path in tqdm(images):
            sample_file = image_path.replace(".jpg", "")
            annot_file = sample_file + ".json"
            img_file = sample_file + ".jpg"
            seg_file = sample_file + ".seg.png"
            with open(annot_file, "r") as f:
                annotation = json.load(f)
            image = Image.open(image_path)
            seg_mask = Image.open(seg_file)
            seg_mask = np.array(seg_mask)
            # print(np.unique(seg_mask))
            shutil.copy(image_path, os.path.join(image_copy_dir, str(image_id)+'.jpg'))
            image_info = create_image_info(image_id, str(image_id)+'.jpg', image.size)
            coco_output["images"].append(image_info)

            objects = annotation['objects']
            for o in objects:
                obj_class = o['class']
                """
                025_mug_16k
                """
                obj_id = obj_info[obj_class]
                # print(obj_id)
                bbox = o['bounding_box']
                bbox_top_left = bbox['top_left']
                y1,x1 = bbox_top_left
                bbox_bottom_right = bbox['bottom_right']
                y2,x2 = bbox_bottom_right
                _x1,_y1,_x2,_y2 = int(x1), int(y1), int(x2), int(y2)
                w = _x2-_x1
                h = _y2-_y1
                
                bbox = [x1,y1,w,h]

                new_seg_mask = seg_mask.copy()
                new_seg_mask[new_seg_mask!=obj_id] = 0
                temp_mask = np.zeros_like(new_seg_mask)
                temp_mask[_y1:_y2,_x1:_x2]=new_seg_mask[_y1:_y2,_x1:_x2]
                mask = temp_mask.astype(bool)
                # counts = np.array(np.unique(mask, return_counts=True)[1], dtype=np.uint8)
                counts = np.sum(mask)
                if counts<=5:
                    print("small_mask")
                    continue
                # cv.imwrite(os.path.join(image_copy_dir, str(segmentation_id)+'.png'), mask)
                # polygon = binary_mask_to_polygon(mask)
                category_info = {}
                category_info['iscrowd']=0
                for i in categories:
                    if i['name']==obj_class:
                        id = i['id']
                category_info['id']=id
                annotation_info = create_annotation_info(segmentation_id, image_id, category_info, mask)
                if annotation_info is not None:
                    coco_output["annotations"].append(annotation_info)
                segmentation_id+=1
            image_id+=1


    with open("./{}".format(save_path),"w") as output_json_file:
        json.dump(coco_output, output_json_file, default=fix_type)

if __name__ == "__main__":
    root_dir = "/disk5/data/fat/mixed"
    save_path = "instances_fat_train.json"

    with open('/disk5/data/fat/mixed/kitchen_0/_object_settings.json', "r") as f:
        obj_info = json.load(f)
    
    obj_info = process_obj_info(obj_info)
    main(root_dir, root_dir+"/", "/disk5/data/fat_copy", save_path, obj_info)
    
    
    
