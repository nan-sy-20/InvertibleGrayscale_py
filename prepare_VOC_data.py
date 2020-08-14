import os
import csv
from sklearn.model_selection import train_test_split

data_root = os.path.join('/home/synan/dataset/VOC2012/JPEGImages')
all_list = os.listdir(data_root)
x_train, x_list = train_test_split(all_list, test_size=3367, random_state=123)
x_test, x_eval = train_test_split(x_list, test_size=67, random_state=123)

new_dataroot = 'datasets/VOC2012'
os.makedirs(new_dataroot, exist_ok=True)

train_paths = []
test_paths = []
eval_paths = []
for item in x_train:
    train_paths.append([os.path.join(data_root, item)])
for item in x_test:
    test_paths.append([os.path.join(data_root, item)])
for item in x_eval:
    eval_paths.append([os.path.join(data_root, item)])

with open(os.path.join(new_dataroot, 'train.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(train_paths)

with open(os.path.join(new_dataroot, 'test.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(test_paths)

with open(os.path.join(new_dataroot, 'eval.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(eval_paths)

print(u'\u2764')
with open(os.path.join(new_dataroot, 'train.csv'), 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)

print(u'\u2764')
with open(os.path.join(new_dataroot, 'test.csv'), 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)

print(u'\u2764')
with open(os.path.join(new_dataroot, 'eval.csv'), 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)