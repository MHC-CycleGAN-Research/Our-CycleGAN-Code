import shutil, os
import csv
import random

image_path = './input/fake2real/testA'
image_seg = './input/fake2real/testSegA'
image_seg_original_L = "/mnt/DataDrive/fakeImgs/realData/uw-sinus-surgery-CL/live/labels"
image_seg_original_C = "/mnt/DataDrive/fakeImgs/realData/uw-sinus-surgery-CL/cadaver/labels"

def create_list(foldername, fulldir=True, suffix=".jpg"):
    """

    :param foldername: The full path of the folder.
    :param fulldir: Whether to return the full path or not.
    :param suffix: Filter by suffix.

    :return: The list of filenames in the folder with given suffix.

    """
    file_list_tmp = os.listdir(foldername)
    file_list = []
    if fulldir is True :
        for item in file_list_tmp:
            if item.endswith(suffix):                                                              
                file_list.append(os.path.join(foldername, item))
    else:
        for item in file_list_tmp:
            if item.endswith(suffix):
                file_list.append(item)
    return file_list

def copy_files():

    list_img = create_list(image_path, True,"jpg")
    list_seg = []
    for string in list_img:
        new_string = string.replace("jpg", "png")
        # new_string = new_string.replace(" (copy)", "")
        if "L" in string:
        	new_string = new_string.replace(image_path, image_seg_original_L)
        elif "S" in string:
        	new_string = new_string.replace(image_path, image_seg_original_C)
        else:
        	input("something is wrong with string: ", string)
        list_seg.append(new_string)

    for f in list_seg:
        shutil.copy(f, image_seg)


if __name__ == '__main__':

	copy_files()