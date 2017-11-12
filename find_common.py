import os
import shutil

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


image_train_path = 'images'


for item in os.listdir(image_train_path):
	ensure_dir('/mnt/zyhu/common/images/' + item)
	ensure_dir('/mnt/zyhu/common/texts/' + item)


def gather_common(src_one, src_two, dst_one, dst_two):
	count = 0
	for item in os.listdir(src_one):
		src_one_subdir  = src_one + '/' + item
		dst_one_subdir  = dst_one + '/' + item

		src_two_subdir = src_two + '/' + item
		dst_two_subdir = dst_two + '/' + item

		one_f_list = [f.split('.')[0] for f in os.listdir(src_one_subdir)]
		two_f_list = [f.split('.')[0] for f in os.listdir(src_two_subdir)]
		common_f_list = list(set(one_f_list).intersection(two_f_list))
		count += 1
		print(count)

		for f in common_f_list:
			f_one_path = src_one_subdir + '/' + f +'.jpg'
			f_two_path = src_two_subdir + '/' + f + '.txt'
			shutil.move(f_one_path,dst_one_subdir)
			shutil.move(f_two_path,dst_two_subdir)

gather_common('/mnt/zyhu/images', '/mnt/zyhu/texts_txt' \
	,'/mnt/zyhu/common/images','/mnt/zyhu/common/texts')
