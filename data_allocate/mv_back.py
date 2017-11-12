import os
import shutil

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


image_train_path = 'images'


for item in os.listdir(image_train_path):
	ensure_dir('/mnt/zyhu/common/images/' + item)
	ensure_dir('/mnt/zyhu/common/texts/' + item)


def mv_back(src_one, src_two, dst_one, dst_two):

	for item in os.listdir(dst_one):
		dst_one_subdir  = dst_one + '/' + item
		src_one_val_subdir  = src_one + '/val/' + item
		src_one_test_subdir  = src_one + '/test/' + item
		src_one_train_subdir  = src_one + '/train/' + item

		dst_two_subdir = dst_two + '/' + item
		src_two_val_subdir  = src_two + '/val/' + item
		src_two_test_subdir  = src_two + '/test/' + item
		src_two_train_subdir  = src_two + '/train/' + item

		for f in os.listdir(src_one_train_subdir):
			f = src_one_train_subdir + '/' + f
			shutil.move(f,dst_one_subdir)

		for f in os.listdir(src_one_test_subdir):
			f = src_one_test_subdir + '/' + f
			shutil.move(f,dst_one_subdir)

		for f in os.listdir(src_one_val_subdir):
			f = src_one_val_subdir + '/' + f
			shutil.move(f,dst_one_subdir)

		for f in os.listdir(src_two_train_subdir):
			f = src_two_train_subdir + '/' + f
			shutil.move(f,dst_two_subdir)

		for f in os.listdir(src_two_test_subdir):
			f = src_two_test_subdir + '/' + f
			shutil.move(f,dst_two_subdir)

		for f in os.listdir(src_two_val_subdir):
			f = src_two_val_subdir + '/' + f
			shutil.move(f,dst_two_subdir)

dst_one = '/mnt/zyhu/common/images'
dst_two = '/mnt/zyhu/common/texts'
src_one = '/mnt/zyhu/common/tmp/images'
src_two = '/mnt/zyhu/common/tmp/texts'

mv_back(src_one, src_two, dst_one, dst_two)

