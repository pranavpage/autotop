# Script to generate data for training 
# Segmentation images and label masks for buildings and roads


# look at functions from create_spacenet_masks.py and apls_utils, write your own
import os, re
def create_building_masks():
    return

def batch_convert_to_8bit(path_raw_tifs, path_tifs):
    # convert all raw images in path_raw_tifs and store their 8 bit versions in path_tifs
    
    return
def create_road_masks(path_tifs, path_geojson_roads, buffer_meters, burn_val,
                      path_root=None, overwrite=True, num_images = 1):
    # with the 8 bit .tif files in path_tifs,  
    # create road masks with the shapes in the directory path_geojson_roads
    # the data path is in path_root
    if(not path_root):
        path_root = os.getcwd()
    output_img_dir = f"{path_root}/road/img"
    output_mask_dir = f"{path_root}/road/label"
    raw_8bit_img_list = os.listdir(path_tifs)
    print(f"Raw image filenames (8 bit) = {raw_8bit_img_list}")
    return
def extract_building_train_data():
    return

def extract_road_train_data():
    return 

def main():
    return 

if(__name__ == "__main__"):
    main()