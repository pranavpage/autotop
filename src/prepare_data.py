# Script to generate data for training 
# Segmentation images and label masks for buildings and roads


# look at functions from create_spacenet_masks.py and apls_utils, write your own
import os, re, rasterio, glob, time
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt 
from rasterio.warp import transform_geom
from rasterio.features import rasterize


def batch_convert_to_8bit(path_raw_tifs, path_tifs, overwrite=True, percs = [2,98]):
    # convert all raw images in path_raw_tifs and store their 8 bit versions in path_tifs
    raw_img_list = sorted(os.listdir(path_raw_tifs))
    cnt = 0
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
                    cnt +=1
    print(f"Batch converted {cnt} images, img list size = {len(raw_img_list)}")
    return
def get_building_im_masks(im_path, gj_path, im_num, out_img_dir, out_mask_dir,
                       burn_val, gdf_crs):
    # Open image, create buffer around road, burn shape to image bounds 
    # store image and mask in output
    gdf = gpd.read_file(gj_path)
    gdf_utm = gdf.to_crs(gdf_crs)
    # shapes ready
    with rasterio.open(im_path) as im:
        if(not gdf_utm.empty):
            transformed_geometries = [transform_geom(gdf_utm.crs, im.crs, shape) 
                                    for shape in gdf_utm.geometry]
            burned_polygons = rasterize(
            [(geom, burn_val) for geom in transformed_geometries],
            out_shape=im.shape,
            transform=im.transform,
            fill=0,
            dtype=rasterio.uint8
            )
        else:
            burned_polygons = np.zeros(im.shape)
        # save image
        output_img_path = f"{out_img_dir}/img{im_num}.tif"
    # save mask 
    output_mask_path = f"{out_mask_dir}/img{im_num}.png"
    plt.imsave(output_mask_path, burned_polygons, cmap='gray', format='png')

    # save image
    output_img_path = f"{out_img_dir}/img{im_num}.tif"
    os.system(f"cp {im_path} {output_img_path}")
    return

def get_road_im_masks(im_path, gj_path, im_num, out_img_dir, out_mask_dir,
                      buffer_meters, burn_val, gdf_crs):
    # Open image, create buffer around road, burn shape to image bounds 
    # store image and mask in output
    gdf = gpd.read_file(gj_path)
    gdf_utm = gdf.to_crs(gdf_crs)
    gdf_utm_buffered = gdf_utm.copy()
    gdf_utm_buffered['geometry'] = gdf_utm.buffer(buffer_meters, cap_style = 1) 
    gdf_utm_dissolve = gdf_utm_buffered.dissolve(by='road_type')
    # shapes ready
    with rasterio.open(im_path) as im:
        transformed_geometries = [transform_geom(gdf_utm_dissolve.crs, im.crs, shape) 
                                  for shape in gdf_utm_dissolve.geometry]
        burned_polygons = rasterize(
        [(geom, burn_val) for geom in transformed_geometries],
        out_shape=im.shape,
        transform=im.transform,
        fill=0,
        dtype=rasterio.uint8
    )
        # save image
        output_img_path = f"{out_img_dir}/img{im_num}.tif"
    # save mask 
    output_mask_path = f"{out_mask_dir}/img{im_num}.png"
    plt.imsave(output_mask_path, burned_polygons, cmap='gray', format='png')

    # save image
    output_img_path = f"{out_img_dir}/img{im_num}.tif"
    os.system(f"cp {im_path} {output_img_path}")
    return


def get_im_num(im_name):
    p = re.compile("(?<=img)\d+")
    res = p.search(im_name)
    im_num = res.group(0)
    return im_num

def plot_polygons_with_extent(gdf, idx=0):
    # Check if the input is a GeoDataFrame
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise ValueError("Input must be a GeoDataFrame.")

    # Plot all the polygons in the GeoDataFrame
    ax = gdf.plot(marker='o', markersize=5, color='k', edgecolor = 'g')

    # Get the extent of the GeoDataFrame
    xmin, ymin, xmax, ymax = gdf.total_bounds

    # Add a rectangle to show the extent
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                               linewidth=2, edgecolor='red', facecolor='none'))

    # Add labels, title, etc., as needed
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Polygons Visualization with Extent")
    plt.show()
    # Show the plot

def create_building_masks(path_tifs, path_geojson_buildings, burn_val=255,
                      path_root=None, overwrite=True, num_images = 1, gdf_crs=27561,
                      gj_file_pattern="buildings_AOI_3_Paris_img{}.geojson"):
    # with the 8 bit .tif files in path_tifs,  
    # create building masks with the shapes in the directory path_geojson_buildings
    # the data path is in path_root
    if(not path_root):
        path_root = os.getcwd()
    output_img_dir = f"{path_root}/data/building/img"
    output_mask_dir = f"{path_root}/data/building/label"
    if(not os.path.isdir(output_img_dir)):
        os.makedirs(output_img_dir)
    if(not os.path.isdir(output_mask_dir)):
        os.makedirs(output_mask_dir)
    raw_8bit_img_list = sorted(os.listdir(path_tifs))[:num_images]
    # print(f"Raw image filenames (8 bit) = {raw_8bit_img_list}")
    cnt = 0
    for im_name in raw_8bit_img_list:

        im_path = f"{path_tifs}/{im_name}"
        im_num = get_im_num(im_name)

        gj_fname = gj_file_pattern.format(im_num)
        gj_path = f"{path_geojson_buildings}/{gj_fname}"
        if(os.path.isfile(gj_path)):
            get_building_im_masks(im_path, gj_path, im_num, out_img_dir=output_img_dir, 
                              out_mask_dir=output_mask_dir,
                             burn_val=burn_val, gdf_crs=gdf_crs)
        cnt+=1
        print(f"num_images = {cnt}/{len(raw_8bit_img_list)}", end="\r")
    print(f"Total images processed = {cnt}")
    return

def create_road_masks(path_tifs, path_geojson_roads, buffer_meters=2, burn_val=255,
                      path_root=None, overwrite=True, num_images = 1, gdf_crs=27561,
                      gj_file_pattern="SN3_roads_train_AOI_3_Paris_geojson_roads_img{}.geojson"):
    # with the 8 bit .tif files in path_tifs,  
    # create road masks with the shapes in the directory path_geojson_roads
    # the data path is in path_root
    if(not path_root):
        path_root = os.getcwd()
    output_img_dir = f"{path_root}/data/road/img"
    output_mask_dir = f"{path_root}/data/road/label"
    if(not os.path.isdir(output_img_dir)):
        os.makedirs(output_img_dir)
    if(not os.path.isdir(output_mask_dir)):
        os.makedirs(output_mask_dir)
    raw_8bit_img_list = sorted(os.listdir(path_tifs))[:num_images]
    # print(f"Raw image filenames (8 bit) = {raw_8bit_img_list}")
    cnt = 0
    for im_name in raw_8bit_img_list:

        im_path = f"{path_tifs}/{im_name}"
        im_num = get_im_num(im_name)

        gj_fname = gj_file_pattern.format(im_num)
        gj_path = f"{path_geojson_roads}/{gj_fname}"
        if(os.path.isfile(gj_path)):
            get_road_im_masks(im_path, gj_path, im_num, out_img_dir=output_img_dir, 
                              out_mask_dir=output_mask_dir,
                              buffer_meters=buffer_meters, burn_val=burn_val, gdf_crs=gdf_crs)
        cnt+=1
        print(f"num_images = {cnt}/{len(raw_8bit_img_list)}", end="\r")
    print(f"Total images processed = {cnt}")
    return
def extract_building_train_data():
    print("Extracting training data for buildings")
    root_dir = os.getcwd()
    print(f"Root dir = {root_dir}")
    path_raw_tifs = f"{root_dir}/raw/buildings/AOI_3_Paris_Train/RGB-PanSharpen"
    path_tifs = f"{root_dir}/raw/buildings/AOI_3_Paris_Train/RGB-PanSharpen_8bit"
    path_geojson_buildings = f"{root_dir}/raw/buildings/AOI_3_Paris_Train/geojson/buildings"
    batch_convert_to_8bit(path_raw_tifs, path_tifs, overwrite=False)
    create_building_masks(path_tifs, path_geojson_buildings, num_images=-1)
    return

def extract_road_train_data():
    print("Extracting training data for roads")
    root_dir = os.getcwd()
    print(f"Root dir = {root_dir}")
    path_raw_tifs = f"{root_dir}/raw/roads/data/Paris/images/AOI_3_Paris/PS-RGB"
    path_tifs = f"{root_dir}/raw/roads/data/Paris/images/AOI_3_Paris/PS-RGB_8bit"
    path_geojson_roads = f"{root_dir}/raw/roads/data/Paris/images/AOI_3_Paris/geojson_roads"
    batch_convert_to_8bit(path_raw_tifs, path_tifs, overwrite=False)
    create_road_masks(path_tifs, path_geojson_roads, num_images=-1)
    return 

def main():
    extract_road_train_data()
    extract_building_train_data()
    return 

if(__name__ == "__main__"):
    main()