import os
import shutil

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


image_train_path = 'images'


for item in os.listdir(image_train_path):
	ensure_dir('/mnt/zyhu/common/images/' + item)
	ensure_dir('/mnt/zyhu/common/texts/' + item)


def split_dataset(src_one, src_two, dst_one, dst_two):

	for item in os.listdir(src_one):
		src_one_subdir  = src_one + '/' + item
		dst_one_val_subdir  = dst_one + '/val/' + item
		dst_one_test_subdir  = dst_one + '/test/' + item

		src_two_subdir = src_two + '/' + item
		dst_two_val_subdir  = dst_two + '/val/' + item
		dst_two_test_subdir  = dst_two + '/test/' + item

		test_one =  int(0.3 * len(os.listdir(src_one_subdir)))
		test_two =  int(0.3 * len(os.listdir(src_two_subdir)))
		if test_two != test_one:
			return

		file_name = [f.split('.')[0] for f in os.listdir(src_one_subdir)]

		val = int(test_two / 2)

		for i in range(0, val):
			f1 = src_one_subdir + '/' + file_name[i] + '.jpg'
			f2 = src_two_subdir + '/' + file_name[i] + '.txt'
			shutil.move(f1, dst_one_val_subdir)
			shutil.move(f2, dst_two_val_subdir)
		for i in range(val, test_one):
			f1 = src_one_subdir + '/' + file_name[i] + '.jpg'
			f2 = src_two_subdir + '/' + file_name[i] + '.txt'
			shutil.move(f1, dst_one_test_subdir)
			shutil.move(f2, dst_two_test_subdir)

def train_dataset(src_one, src_two, dst_one, dst_two):

	for item in os.listdir(src_one):
		src_one_subdir  = src_one + '/' + item
		dst_one_train_subdir  = dst_one + '/train/' + item

		src_two_subdir = src_two + '/' + item
		dst_two_train_subdir  = dst_two + '/train/' + item

		file_name = [f.split('.')[0] for f in os.listdir(src_one_subdir)]

		for f in file_name:
			shutil.move(src_one_subdir + '/' + f + '.jpg',dst_one_train_subdir)
			shutil.move(src_two_subdir + '/' + f + '.txt',dst_two_train_subdir)


src_one = '/mnt/zyhu/common/images'
src_two = '/mnt/zyhu/common/texts'
dst_one = '/mnt/zyhu/common/tmp/images'
dst_two = '/mnt/zyhu/common/tmp/texts'

split_dataset(src_one, src_two, dst_one, dst_two)
train_dataset(src_one, src_two, dst_one, dst_two)

