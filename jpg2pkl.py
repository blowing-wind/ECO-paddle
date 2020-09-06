import os
import pickle
import numpy as np

def get_list(path):
    with open(path) as f:
        lines = f.read().splitlines()
    vids = [line.split()[0].split('/')[-1].replace('.avi', '') for line in lines]
    return set(vids)

train_list = get_list('split01_ucf/trainlist01.txt')
val_list = get_list('split01_ucf/testlist01.txt')

label_dic = np.load('label_dir.npy', allow_pickle=True).item()

source_dir = 'data/UCF-101'
target_train_dir = 'data/UCF-101/train'
target_val_dir = 'data/UCF-101/val'

if not os.path.exists(target_train_dir):
    os.mkdir(target_train_dir)
if not os.path.exists(target_val_dir):
    os.mkdir(target_val_dir)

for idx, key in enumerate(label_dic):
    class_jpg = key + '_jpg'

    class_dir = os.path.join(source_dir, class_jpg)
    videos = os.listdir(class_dir)
    for vid in videos:
        images = os.listdir(os.path.join(class_dir, vid))
        images.sort()
        frames = [os.path.join(class_dir, vid, f) for f in images]
        output_pkl = vid + '.pkl'
        if vid in train_list:
            output_pkl = os.path.join(target_train_dir, output_pkl)
        elif vid in val_list:
            output_pkl = os.path.join(target_val_dir, output_pkl)
        else:
            raise FileNotFoundError('{} not in train or val list!'.format(vid))
        with open(output_pkl, 'wb') as f:
            pickle.dump((vid, label_dic[key], frames), f, -1)
    print('class {} done. {}/{}'.format(key, idx, len(label_dic)))
