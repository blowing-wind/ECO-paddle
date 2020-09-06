import os
import numpy as np
import cv2
import threading

src_path = 'data/UCF-101'
label_name = os.listdir(src_path)
# label_dir = {}
NUM_THREADS = 10

def split(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def extract(video, class_src_path):
    class_save_path = class_src_path + '_jpg'
    vid, _ = video.split('.')
    video_save_path = os.path.join(class_save_path, vid)
    if not os.path.exists(video_save_path):
        os.mkdir(video_save_path)
    
    cap = cv2.VideoCapture(os.path.join(class_src_path, video))
    frame_count = 1
    success = True
    while success:
        success, frame = cap.read()
        # print('read a new frame:', success)
        params = []
        params.append(1)
        if success:
            cv2.imwrite(os.path.join(video_save_path, '%05d.jpg'%frame_count), frame, params)
        frame_count += 1
    cap.release()

def target(videos, class_src_path, idx):
    for vid in videos:
        extract(vid, class_src_path)
    print('{} class split {} extract finished.'.format(class_src_path.split('/')[-1], idx))

for index, label in enumerate(label_name):
    if label.startswith('.'):
        continue
    # label_dir[label] = index
    class_src_path = os.path.join(src_path, label)
    class_save_path = os.path.join(src_path, label) + '_jpg'
    if not os.path.exists(class_save_path):
        os.mkdir(class_save_path)

    videos = os.listdir(class_src_path)

    splits = list(split(videos, NUM_THREADS))
    threads = []
    for i, s in enumerate(splits):
        thread = threading.Thread(target=target, args=(s, class_src_path, i))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

#np.save('label_dir.npy', label_dir)
#print(label_dir)
