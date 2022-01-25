from pycocotools.coco import COCO
import os
import shutil
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--imgspath', type=str, default='./data/raw_images')
parser.add_argument('--annostrain', type=str, default='./data/train.json')
parser.add_argument('--annostest', type=str, default='./data/test.json')
args = parser.parse_args()
dataset_type = ['train', 'val']


def save_yolo_format(imgs_path, annotation_file, dataset_type):
    coco = COCO(annotation_file=annotation_file)
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    cat_ids = coco.getCatIds(catNms=nms)
    images = coco.loadImgs(coco.getImgIds())

    if os.path.exists('./data/images/' + dataset_type):
        shutil.rmtree('./data/images/' + dataset_type)
    os.makedirs('./data/images/' + dataset_type)

    if os.path.exists('./data/labels/' + dataset_type):
        shutil.rmtree('./data/labels/' + dataset_type)
    os.makedirs('./data/labels/' + dataset_type)

    for image in tqdm(images):
        annos_list = []
        dw = 1. / image['width']
        dh = 1. / image['height']

        anno_ids = coco.getAnnIds(
            imgIds=image['id'], catIds=cat_ids, iscrowd=None)
        annos = coco.loadAnns(ids=anno_ids)
        shutil.copy(src=imgs_path + '/' + image['file_name'],
                    dst='./data/images/' + dataset_type + '/' + image['file_name'])

        file_name = image['file_name'].replace('.jpg', '.txt')

        for i in range(len(annos)):
            xmin = annos[i]["bbox"][0]
            ymin = annos[i]["bbox"][1]
            width = annos[i]["bbox"][2]
            height = annos[i]["bbox"][3]

            x_center = xmin + width/2
            y_center = ymin + height/2

            x_center = x_center * dw
            width = width * dw
            y_center = y_center * dh
            height = height * dh

            annos_list.append("{} {:.5f} {:.5f} {:.5f} {:.5f}".format(
                annos[i]['category_id']-1, x_center, y_center, width, height))
        save_file_name = './data/labels/' + dataset_type + '/' + file_name
        print("\n".join(annos_list), file=open(save_file_name, "w"))


if __name__ == '__main__':
    print('Start convert yolo format')
    save_yolo_format(args.imgspath, args.annostrain, dataset_type[0])
    save_yolo_format(args.imgspath, args.annostest, dataset_type[1])
    print('Done')
