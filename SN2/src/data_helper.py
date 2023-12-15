import numpy as np
import os, re, shutil
from skimage.io import imread, imsave
data_path = "/mnt/l1/auto_top/SN2/data/AOI_3_Paris_Train"
seg_data_path = f"{data_path}/seg_data"
seg_test_data_path = f"{data_path}/seg_test_data"
target_img_path = f"{data_path}/patch/img"
target_mask_path = f"{data_path}/label/img"
og_img_path = f"{data_path}/RGB-PanSharpen_8bit"
og_mask_path = f"{data_path}/masks_2m"

# convert each img .tif file to a .png file, save with img<im_num>.png in patch/img/
# save mask with img<im_num>.png in label/img/

# p = re.compile("(?<=img)\d+")
#     res = p.search(fname)
#     im_num = res.group(0)

im_list = sorted(os.listdir(f"{og_img_path}/"))
print(f"Images = {len(im_list)}, {im_list[:2]}")

mask_list = sorted(os.listdir(f"{og_mask_path}/"))
print(f"masks = {len(mask_list)}, {mask_list[:2]}")
cnt = 0
test_split = 0.1
m_thresh = 1e-4
test_cnt=0
for im_name in im_list:
    p = re.compile(r"(?<=img)\d+")
    res = p.search(im_name)
    im_num = res.group(0)
    
    mask_name = f"RGB-PanSharpen_AOI_3_Paris_mask_img{im_num}.png"

    im_path = f"{og_img_path}/{im_name}"
    mas_path = f"{og_mask_path}/{mask_name}"

    target_im_path = f"{target_img_path}/img{im_num}.png"
    target_mas_path = f"{target_mask_path}/img{im_num}.png"
    im = imread(im_path)
    mas = imread(mas_path, as_gray=True)
    m_cover = np.average(mas)/255.0
    if(m_cover<m_thresh):
        print(f"Dropping im_num={im_num}, m_cover = {m_cover*100:2.2f}%")
    else:
        cnt +=1
        
        if(np.random.rand()<test_split):
            imsave(f"{seg_test_data_path}/patch/img/img{im_num}.png", im)
            shutil.copy(mas_path, f"{seg_test_data_path}/label/img/img{im_num}.png")
            test_cnt+=1
        else:
            imsave(f"{seg_data_path}/patch/img/img{im_num}.png", im)
            shutil.copy(mas_path, f"{seg_data_path}/label/img/img{im_num}.png")
print(f"Total pairs of (650, 650) size = {cnt}, test={test_cnt}")