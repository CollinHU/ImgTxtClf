import os
import shutil

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


image_train_path = 'images/train'
image_test_path = 'images/test'


for item in os.listdir(image_train_path):
	ensure_dir('/mnt/zyhu/images/' + item)


def move_file(source, destination):
	for item in os.listdir(source):
		source_subdir  = source + '/' + item
		destination_subdir = destination + '/' + item
		for file in os.listdir(source_subdir):
			f_path = source_subdir + '/' + file
			shutil.move(f_path, destination_subdir)

#move_file(image_test_path, '/mnt/zyhu/images')
move_file(image_train_path, '/mnt/zyhu/images')

