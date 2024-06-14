import pandas as pd
import json
import os


def relative_to_absolute(rel_x, rel_y, width, height, img_width, img_height) -> list:
    x = rel_x * img_width
    y = rel_y * img_height
    w = width * img_width
    h = height * img_height
    return [x, y, w, h]


def convert_csv_to_coco(dataframe: pd.DataFrame, output_file: str) -> None:
    dataframe = dataframe.dropna()

    coco_format = {
        'images': [],
        'annotations': [],
        'categories': []
    }

    annotation_id = 1
    image_id = 1

    classes = dataframe['class'].unique()
    for i, cls in enumerate(classes):
        coco_format['categories'].append({
            'id': int(cls),
            'name': f'class{int(cls)}',
            'supercategory': 'none'
        })

    image_info = {}

    for index, row in dataframe.iterrows():
        file_path = row['jpg_file']
        cls = row['class']
        rel_x = row['x']
        rel_y = row['y']
        width = row['w']
        height = row['h']

        if file_path not in image_info:
            img_width, img_height = 2160, 3840
            image_info[file_path] = (image_id, img_width, img_height)
            coco_format['images'].append({
                'id': image_id,
                'file_name': os.path.basename(file_path),
                'width': img_width,
                'height': img_height
            })
            image_id += 1
            current_image_id = image_id
        else:
            current_image_id, img_width, img_height = image_info[file_path]

        bbox = relative_to_absolute(rel_x, rel_y, width, height, img_width, img_height)

        coco_format['annotations'].append({
            'id': annotation_id,
            'image_id': current_image_id,
            'category_id': int(cls),
            'bbox': bbox,
            'area': bbox[2] * bbox[3],
            'segmentation': [],
            'iscrowd': 0
        })

        annotation_id += 1

    with open(output_file, 'w') as f:
        json.dump(coco_format, f)

