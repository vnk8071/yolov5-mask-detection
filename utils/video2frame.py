import cv2
import os
import argparse
import pandas as pd
import sys


def video_to_frames(video, dataset_name, output):
    # create output dir if it does not exist
    if not os.path.exists(output):
        os.mkdir(output)

    print('Video: ', video)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Total frame: {total_frames} | fps: {fps}')
    print('Extracting frame ...', end='')
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            count += 1
            video = video.replace('+', '_')
            filename = os.path.join(output, dataset_name.replace(' ', '_') + '_' + video.split(
                '/')[-1].split('.')[0] + '_{}'.format(count) + '.jpg')
            cv2.imwrite(filename, frame)
        else:
            cap.release()
            break

    assert total_frames == count, 'Frames not extracted properly'
    print(f'Done | {count} frames extracted')
    print('Saved frames are in ', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='DryEye Video path',
                        default='./data/videos_mp4')
    parser.add_argument('--annotation_path', help='LabelBox annotation paths', nargs='+',
                        default=['./data/raw_labels_gs1.csv', './data/raw_labels.csv'])
    parser.add_argument('--output', help='output directory',
                        default='./data/raw_images')
    args = parser.parse_args()
    annotation_path = args.annotation_path
    data_path = args.data_path
    output = args.output

    df = [pd.read_csv(anno_path) for anno_path in annotation_path]
    df = pd.concat(df, ignore_index=True)
    print(df[['External ID', 'Dataset Name']])

    if os.path.isdir(data_path):
        for i, filename in enumerate(os.listdir(data_path)):
            dataset_name = df['Dataset Name'].iloc[df.index[filename == df['External ID']].tolist()[
                0]]
            print(f'\n### {i+1}/{len(os.listdir(data_path))}')
            video_to_frames(os.path.join(data_path, filename),
                            dataset_name, output)
    else:  # is a video file
        video_to_frames(data_path, None, output)
