import argparse
import pandas as pd
import numpy as np
import json
import os
from datetime import date
from dryeye_utils import LabelBox
from dryeye_utils import VideoProcessing
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

today = date.today()
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJja21iZ3FpbW80bHpsMDc4OXVxamxvcXdxIiwib3JnYW5pemF0aW9uSWQiOiJja21iZ3Fpa3VmcGVmMDg0OG0yZ2M5NDkxIiwiYXBpS2V5SWQiOiJja21ibnIzOW0zcXNpMDcwOWxnMWh3dmloIiwiaWF0IjoxNjE1ODc3NTE2LCJleHAiOjIyNDcwMjk1MTZ9.5Ntv6fXtcFvjM5imNpE2kL6pfRhXkps9NTN-WpPLKCo"

COCO_TEMPLATE = {
    "info": {},
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [],
    "segment_info": []
}


class Dataset:
    def __init__(self, data_path, annotation_path, **args):
        '''
        Default
        Input:
        - name: dataset name
        - annotation: annotation file in json
        - data_path: data folder
        '''
        self.data_path = data_path
        self.annotation_path = annotation_path
        for key in args:
            setattr(self, key, args[key])

    def __load__(self):
        '''
        Load annotations from JSON file
        '''
        with open(self.annotation_path) as f:
            self.anns_json = json.load(f)
            return self

    def __get_category_name__(self, key):
        '''
        Convert decimal key to lesions combinations
        '''
        self.__load__()
        categories = self.anns_json['categories']
        name = ''
        for idx, num_str in enumerate(bin(key)[:1:-1]):
            if int(num_str) == 1:
                combination_str = categories[idx]['name']
                name = '+'.join([name, combination_str])
        name = name[1:]
        return name

    def __bin_to_dec__(self, stats):
        '''
        Convert lesions binary list (LSB first) to decimal represented lesions combinations
        '''
        result = np.zeros((len(stats)), dtype=int)
        for idx, row in enumerate(stats):
            for order, element in enumerate(row):
                result[idx] += element * (2 ** order)
        return result

    def histogram(self, data, titles=None):
        '''
        Plot histogram
        '''
        fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
        axs.hist(data, bins=np.arange(data.min(), data.max()+1))
        if titles is not None and len(titles) == 3:
            axs.set_title(titles[0])
            axs.set_xlabel(titles[1])
            axs.set_ylabel(titles[2])

    def stats(self):
        '''
        General dataset statistics
        '''
        raise NotImplementedError


class DryEyeDataset(Dataset):
    '''
    Dry Eye Golden II Dataset
    Input:
        - data_path: path to videos
        - annotation_path: path to annotation JSON file
    Output:
        - coco_annoatations.json
    '''

    def __init__(self, **args):
        super(DryEyeDataset, self).__init__(**args)

    def __get_dict(self):
        '''
        Load annotations from JSON file
        '''
        with open(self.annotation_path) as f:
            anns = json.load(f)
            return anns

    def preprocess(self):
        print('Retrieve annotation from LabelBox')
        metadata = [pd.read_csv(anno_path) for anno_path in annotation_path]
        metadata = pd.concat(metadata, ignore_index=True)
        labelbox = LabelBox(api_key=API_KEY)
        labelbox_objects = []
        for index, row in metadata.iterrows():
            video_name = row["External ID"]
            dataset_name = row["Dataset Name"]
            label = row["Label"]
            url = json.loads(label).get('frames')
            objects = labelbox.get_labelbox_annotation(
                url=url).get_object(dataset_name, video_name)
            if objects is None:
                continue
            labelbox_objects.extend(objects)
        print('Get annotation from Labelbox successfully')
        self.data_frame = pd.DataFrame(labelbox_objects)
        return self

    def convert_to_coco(self, is_train=False, version='0', output='./data'):
        print('Convert to COCO format')
        # Get filename
        df = self.data_frame
        df.to_csv(os.path.join(output, 'labelbox_objects_' + today.strftime("%m%d%Y") + '_v' +
                  version + '.csv'), index=False)

        # Load categories
        with open('./utils/categories_train.json', 'r') as f:
            COCO_TEMPLATE['categories'] = json.load(f)
        classes = [name['name'] for name in COCO_TEMPLATE['categories']]
        classes_id = [id for id in range(len(classes))]
        df_category = pd.DataFrame(
            {'object': classes, 'category_id': classes_id})

        assert len(list(set(df['object'].tolist()).difference(
            classes))) == 0, "Should use --is-train because got new classes"

        df = df.merge(df_category, on="object", how="left", sort=False)
        file_name = df.apply(lambda x: x['dataset'].replace(
            ' ', '_') + '_' + x['video'][:-4].replace('+', '_') + '_' + str(x['frame_id']) + '.mp4', axis=1)
        df['file_name'] = file_name
        df = df.assign(image_id=df['file_name'].astype('category').cat.codes)

        if is_train:
            df = df.assign(category_id=df['object'].astype(
                'category').cat.codes)
            categories = df.iloc[df['category_id'].drop_duplicates(
            ).index][['object', 'category_id']].sort_values(by=['category_id'])

            COCO_TEMPLATE['categories'] = [{
                "supercategory": x[0],
                "id": x[1] + 1,
                "name": x[0]
            } for x in categories.to_numpy()]

            # Save categories train config
            with open('./tbupdetector/categories_train.json', 'w') as f:
                json.dump(COCO_TEMPLATE['categories'], f, indent=4)

        COCO_TEMPLATE['info'] = {
            "description": df['dataset'].iloc[0],
            "url": '',
            "version": version,
            "year": 2021,
            "contributor": "MD.Yokoi",
            "date_created": "2021"
        }

        COCO_TEMPLATE['licenses'] = [
            {
                "url": "http://a2ds.io",
                "id": 0,
                "name": "A2DS"
            }
        ]

        video_root = self.data_path
        video = VideoProcessing()
        images = []

        file_name_df = df[['file_name', 'video']].sort_values(
            by=['file_name']).drop_duplicates(keep='first').reset_index()
        for idx, row in file_name_df.iterrows():
            images.append({
                "id": idx + 1,
                "license": 0,
                "file_name": row['file_name'].replace('.mp4', '.jpg'),
                "coco_url": "",
                "height": int(video.dimension(os.path.join(video_root, row['video']))[1]),
                "width": int(video.dimension(os.path.join(video_root, row['video']))[0]),
                "date_captured": "2021",
                "flickr_url": "",
            })
        COCO_TEMPLATE['images'] = images

        annotations = []

        for idx, row in df.iterrows():
            annotations.append({
                "id": idx + 1,
                "image_id": row['image_id'] + 1,
                "iscrowd": 0,
                "category_id": row['category_id'] + 1,
                "segmentation": [],
                "bbox": [row['bbox']['left'], row['bbox']['top'], row['bbox']['width'], row['bbox']['height']],
                "area": row['bbox']['height'] * row['bbox']['width'],
                "bbox_mode": 1
            })
            COCO_TEMPLATE['annotations'] = annotations
        print('File includes {} object with annotation'.format(
            len(COCO_TEMPLATE['annotations'])))
        save_filename = '{}/coco_annotations_{}_v{}.json'.format(
            output, today.strftime("%m%d%Y"), version)
        with open(save_filename, 'w') as f:
            json.dump(COCO_TEMPLATE, f, indent=4)
        print('Please check output file at ', save_filename)

    def stats(self):
        '''
        General dataset statistics
        '''
        self.__load__()
        stats = {}
        images = self.anns_json['images']
        annotations = self.anns_json['annotations']
        categories = self.anns_json['categories']
        # Number of images:
        # print('Number of images: {}'.format(len(images)))
        # Number of objects per category
        # print('Number of categories: {}'.format(len(categories)))
        df = pd.DataFrame(annotations)
        objects_count = df['category_id'].value_counts()
        objects = []
        for idx, obj in enumerate(objects_count):
            objects.append({
                categories[idx]['name']: obj
            })
        # Number of images contains specific objects
        cat_by_img = df[['image_id', 'category_id']]
        cat_by_img = cat_by_img.pivot_table(index='image_id', columns='category_id',
                                            aggfunc=len, fill_value=0)
        cat_by_img = cat_by_img.where(cat_by_img == 0, 1)
        # print(cat_by_img.value_counts())
        cat_by_img = cat_by_img.to_numpy()

        combination = self.__bin_to_dec__(cat_by_img)
        unique, counts = np.unique(combination, return_counts=True)
        combination_list = list(zip([self.__get_category_name__(x) for x in unique], [
            int(count) for count in counts]))
        stats = {
            'images': len(images),
            'categories': len(categories),
            'objects': objects,
            'combination_list': combination_list
        }
        with open('stats.json', 'w') as f:
            json.dump(stats, f)
        return self

    # TODO: this function is not functional. Need more work
    def sampling(self, scale=0.8, threshold=8):
        '''
        Sample dataset by two rules:
            - Keep distribution of lesions combinations
            - Balance amount of lesions per class.
        Input:
            - scale: splitting scale of train_set/test_set
            - threshold: an integer separates a list of images into 2 subgroup: 'few lesions', 'many lesions'
        *threshold = 8 gives the minimum variance of the amount of lesions after folding.
        '''
        tbups_filenames = [x for x in os.listdir(
            '{}/images'.format(self.data_path))]
        tbups_count = np.zeros((len(tbups_filenames), 5), dtype=int)

        anns_coco = self.__get_dict()
        annotations = anns_coco['annotations']
        images = anns_coco['images']
        for idx, annotation in enumerate(annotations):
            bboxs = annotation['bbox']
            image = images[annotation['image_id']-1]
            annotation['file_name'] = os.path.join(
                self.data_path, 'images', image['file_name'].split('/')[-1])
            tbups_count[image['id'], annotation['category_id']-1] += 1

        image_id = [annotation['image_id'] for annotation in annotations]
        count_mask = (tbups_count.sum(axis=1) >= threshold)*1
        ratio = int(scale/(1 - scale)) + 1
        skf = StratifiedKFold(n_splits=ratio)
        folds = []
        for train_index, test_index in skf.split(image_id, count_mask):
            train_set = np.array(annotations)[train_index]
            test_set = np.array(annotations)[test_index]
            folds.append([train_set, test_set])

        return folds


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', help='DryEye Video path',
                    default='./data/videos_mp4')
parser.add_argument('--version', help='Version to save Labelbox object csv format',
                    default='0')
parser.add_argument('--annotation_path', help='LabelBox annotation paths', nargs='+',
                    default=['./data/raw_labels_gs1.csv', './data/raw_labels.csv'])
parser.add_argument(
    '--is_train', help='True if use data for train', action='store_true')
parser.add_argument('--output', help='output directory', default='./data')

args = parser.parse_args()
data_path = args.data_path
version = args.version
annotation_path = args.annotation_path
is_train = args.is_train
output = args.output


if __name__ == '__main__':
    print('Start to process raw annotation')
    dryeye = DryEyeDataset(data_path=data_path,
                           annotation_path=annotation_path)
    dryeye.preprocess().convert_to_coco(is_train, version, output)
    print('Done...')
