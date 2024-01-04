# Script to generate data for training 
# Segmentation images and label masks for buildings and roads


# look at functions from create_spacenet_masks.py and apls_utils, write your own
import os, re, rasterio, glob, time
import numpy as np
def create_building_masks():
    return

def batch_convert_to_8bit(path_raw_tifs, path_tifs, overwrite=True, percs = [2,98]):
    # convert all raw images in path_raw_tifs and store their 8 bit versions in path_tifs
    raw_img_list = sorted(os.listdir(path_raw_tifs))
    for raw_fname in raw_img_list:
        im_raw_path = f"{path_raw_tifs}/{raw_fname}"
        im_path = f"{path_tifs}/{raw_fname}"
        if(not os.path.isfile(im_path) or overwrite):
            # generate file
            with rasterio.open(im_raw_path) as src:
                data = src.read()
                scaled_bands = []
                for band in data:
                    min_value = np.percentile(band, percs[0])
                    max_value = np.percentile(band, percs[1])
                    clipped_band = np.clip(band, min_value, max_value)
                    scaled_band = ((clipped_band - min_value) / (max_value - min_value) * 255).astype(np.uint8)
                    scaled_bands.append(scaled_band)

                scaled_data = np.stack(scaled_bands)

                dst_meta = src.meta.copy()
                dst_meta.update({
                    'count': scaled_data.shape[0],
                    'dtype': 'uint8',
                })

                with rasterio.open(im_path, 'w', **dst_meta) as dst:
                    dst.write(scaled_data)
    return

def get_road_buffer():
    return

def get_im_num(im_name):
    p = re.compile("(?<=img)\d+")
    res = p.search(im_name)
    im_num = res.group(0)
    return im_num

def create_road_masks(path_tifs, path_geojson_roads, buffer_meters=2, burn_val=200,
                      path_root=None, overwrite=True, num_images = 2, 
                      gj_file_pattern="SN3_roads_train_AOI_3_Paris_geojson_roads_img{}.geojson"):
    # with the 8 bit .tif files in path_tifs,  
    # create road masks with the shapes in the directory path_geojson_roads
    # the data path is in path_root
    if(not path_root):
        path_root = os.getcwd()
    output_img_dir = f"{path_root}/road/img"
    output_mask_dir = f"{path_root}/road/label"
    raw_8bit_img_list = sorted(os.listdir(path_tifs))
    print(f"Raw image filenames (8 bit) = {raw_8bit_img_list}")

    for im_name in raw_8bit_img_list[:num_images]:
        print(im_name)
        im_path = f"{path_tifs}/{im_name}"
        im_num = get_im_num(im_name)
        print(f"Imnum = {im_num}")

        gj_fname = gj_file_pattern.format(im_num)
        gj_path = f"{path_geojson_roads}/{gj_fname}"
        if(os.path.isfile(gj_path)):
            print("gj file exists")

    return
def extract_building_train_data():
    return

def extract_road_train_data():
    return 

def main():
    root_dir = os.getcwd()
    print(f"Root dir = {root_dir}")
    path_raw_tifs = f"{root_dir}/raw/roads/data/Paris/images/AOI_3_Paris/PS-RGB"
    path_tifs = f"{root_dir}/raw/roads/data/Paris/images/AOI_3_Paris/PS-RGB_8bit"
    path_geojson_roads = f"{root_dir}/raw/roads/data/Paris/images/AOI_3_Paris/geojson_roads"
    batch_convert_to_8bit(path_raw_tifs, path_tifs, overwrite=False)
    create_road_masks(path_tifs, path_geojson_roads, )
    return 

if(__name__ == "__main__"):
    main()