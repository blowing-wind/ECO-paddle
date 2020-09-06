import os

data_dir = 'data/UCF-101/'

train_data = os.listdir(data_dir + 'train')
print('train data length: ', len(train_data))

val_data = os.listdir(data_dir + 'val')
print('validate data length: ', len(val_data))

with open(data_dir + 'train.list', 'w') as f:
    for line in train_data:
        f.write(data_dir + 'train/' + line + '\n')

with open(data_dir + 'val.list', 'w') as f:
    for line in val_data:
        f.write(data_dir + 'val/' + line + '\n')
