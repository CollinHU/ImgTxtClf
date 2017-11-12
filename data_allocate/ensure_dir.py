import os

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


image_train_path = 'images'


for item in os.listdir(image_train_path):
	ensure_dir('/mnt/zyhu/common/tmp/images/val/' + item)
	ensure_dir('/mnt/zyhu/common/tmp/texts/val/' + item)
