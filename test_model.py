# Script to test trained model
# !pip install git+https://github.com/qubvel/efficientnet
# !pip install git+https://github.com/qubvel/classification_models.git
# !pip install git+https://github.com/qubvel/segmentation_models
# !pip install -U git+https://github.com/albu/albumentations
import tensorflow as tf 
import cv2, os, re, sknw
from osgeo import gdal
os.environ["SM_FRAMEWORK"] = "tf.keras"
import matplotlib.pyplot as plt 
import numpy as np 
from efficientnet.tfkeras import EfficientNetB0
from keras.optimizers import Adam
from skimage.io import imread
from skimage.morphology import skeletonize
from segmentation_models.losses import bce_jaccard_loss, bce_dice_loss
from segmentation_models.metrics import iou_score



def fname_from_fnum(im_num):
    # helper function to return image.tif file path from specified im_num
    # to be fed into pipeline 
    im_path = f"{aoi_path}/PS-RGB_8bit/SN3_roads_train_AOI_3_Paris_PS-RGB_img{im_num}.tif"
    return im_path

def get_img_masks_raw(fname, road_model, building_model):
    # returns image and the masks 
    im = imread(fname)
    print(f"Original Shape: {im.shape}")
    im = im[:,:,:3]

    # Pad the image to make both dimensions divisible by 32
    pad_height = (32 - im.shape[0] % 32) % 32
    pad_width = (32 - im.shape[1] % 32) % 32

    padded_img = cv2.copyMakeBorder(im, pad_height // 2, pad_height - pad_height // 2,
                                    pad_width // 2, pad_width - pad_width // 2,
                                    cv2.BORDER_REFLECT)

    print(f"Padded Shape: {padded_img.shape}")
    w_p, h_p, _ = padded_img.shape
    # TODO : handle any image size
    
    # inference on cropped img 
    road_mask_p = (road_model.predict(np.expand_dims(padded_img, axis=0)/255.0)).reshape(w_p,h_p)
    road_mask_p*=255.0

    building_mask_p = (building_model.predict(np.expand_dims(padded_img, axis=0)/255.0)).reshape(w_p,h_p)
    building_mask_p*=255.0

    # Convert the masks to uint8
    building_mask = building_mask_p.astype(np.uint8)
    road_mask = road_mask_p.astype(np.uint8)
    return padded_img, road_mask, building_mask

def blend_both_masks(img, road_mask, building_mask, alpha=0.3):
    # blend both masks and plot on b/w img
    # Create an empty image for overlaying the masks
    overlay = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    overlay = cv2.merge([overlay, overlay, overlay])
    # Create red mask for buildings
    building_mask = cv2.merge([np.zeros_like(building_mask), np.zeros_like(building_mask), building_mask])

    alpha_building = alpha  # Adjust transparency (alpha) for buildings
    cv2.addWeighted(building_mask, alpha_building, overlay, 1 - alpha_building, 0, overlay)

    # Create green mask for roads with transparency (adjust alpha)
    road_mask = cv2.merge([np.zeros_like(road_mask), road_mask, np.zeros_like(road_mask)])
    alpha_road = alpha  # Adjust transparency (alpha) for roads
    cv2.addWeighted(road_mask, alpha_road, overlay, 1 - alpha_road, 0, overlay)
    return overlay
def refine_mask(mask, binarize_thr = 0.2, do_closing=False, iters = 1, kernel_size = 5):
    # Step 1 : binarize mask
    _, b_mask = cv2.threshold(mask, int(binarize_thr*255), 255, cv2.THRESH_BINARY)
    final_mask = b_mask
    if(do_closing):
        # define the kernel 
        kernel = np.ones((kernel_size, kernel_size), np.uint8) 
    
        # closing the image 
        final_mask = cv2.morphologyEx(b_mask, cv2.MORPH_CLOSE, kernel, iterations=iters)
    else:
        # define the kernel 
        kernel = np.ones((kernel_size, kernel_size), np.uint8) 
    
        # opening the image 
        final_mask = cv2.morphologyEx(b_mask, cv2.MORPH_OPEN, kernel, iterations=iters) 

    return final_mask
def refine_road_mask(road_mask, kernel_size = 7, binarize_thr = 0.4):
    return refine_mask(road_mask, binarize_thr=binarize_thr, do_closing=True, kernel_size=kernel_size)

def refine_building_mask(building_mask, kernel_size = 13, binarize_thr = 0.4):
    return refine_mask(building_mask, binarize_thr=binarize_thr, do_closing=False, kernel_size=kernel_size)

def skeletonize_road_mask(road_mask):
    road_skeleton = np.uint8(skeletonize(road_mask)*255)
    return road_skeleton
def polygonize_building_mask(building_mask):
    contours, _ = cv2.findContours(building_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_polygons_image = np.zeros_like(building_mask, dtype=np.uint8)

    # Iterate through the contours and draw bounding polygons
    for contour in contours:
        # Approximate the contour to get a polygon
        epsilon = 0.04 * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True)

        # Draw the bounding polygon on the image
        cv2.polylines(bounding_polygons_image, [polygon], isClosed=True, color=255, thickness=1)

    print(f"Number of contours = {len(contours)}")
    return bounding_polygons_image
def burn_outlines(img, road_skeleton, building_polygons, dilate_kernel_size = 0):
    road_skeleton_green = cv2.merge([np.zeros_like(road_skeleton), road_skeleton, np.zeros_like(road_skeleton)])
    building_polygons_red = cv2.merge([np.zeros_like(building_polygons), np.zeros_like(building_polygons), building_polygons])
    kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    if(dilate_kernel_size):
        road_skeleton_green = cv2.dilate(road_skeleton_green, kernel, iterations=1)
        building_polygons_red = cv2.dilate(building_polygons_red, kernel, iterations=1)
    ret_img = cv2.add(img, road_skeleton_green)
    ret_img = cv2.add(ret_img, building_polygons_red)
    return ret_img

def get_road_graph(road_skeleton):
    _, b_road_skeleton = cv2.threshold(road_skeleton, 127, 255, cv2.THRESH_BINARY) 
    graph = sknw.build_sknw(b_road_skeleton)
    print(graph)
    # draw image
    plt.imshow(img, cmap='gray')

    # draw edges by pts
    for (s,e) in graph.edges():
        ps = graph[s][e]['pts']
        plt.plot(ps[:,1], ps[:,0], 'green')
    # draw node by o
    nodes = graph.nodes()
    ps = np.array([nodes[i]['o'] for i in nodes])
    plt.plot(ps[:,1], ps[:,0], 'y.')
    plt.show()
    return graph

def plot_before_after(m1, m2, tag="placeholder"):
    plt.subplot(1,2,1)
    plt.imshow(m1, cmap='Greys')
    plt.subplot(1,2,2)
    plt.imshow(m2, cmap='Greys')
    plt.title(tag)
    plt.show()
    return 
def plot_cv2(im):
    cv2.imshow('Image', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 
def print_transform_and_ref(fpath):
    dataset = gdal.Open(fpath)

    # Get the geotransform information
    geotransform = dataset.GetGeoTransform()

    # Print the geotransform information
    print("Geotransform:")
    print("Origin (top left corner):", geotransform[0], geotransform[3])
    print("Pixel Size:", geotransform[1], geotransform[5])
    print("Rotation/Skew:", geotransform[2], geotransform[4])

    # Get the spatial reference information
    spatial_reference = dataset.GetProjection()

    # Print the spatial reference information
    print("\nSpatial Reference:")
    print(spatial_reference)
    return
if(__name__=="__main__"):
    test_patches = False
    test_large = False
    test_both_building_roads = False
    sn3_model_path = "/mnt/l1/auto_top/SN3/models/unet_sn3_high.h5"
    sn2_model_path = "/mnt/l1/auto_top/SN2/models/unet_sn2.h5"

    road_model = tf.keras.models.load_model(sn3_model_path, compile=False)
    building_model = tf.keras.models.load_model(sn2_model_path, compile=False)

    aoi_path = "/mnt/l1/auto_top/SN3/data/Paris/images/AOI_3_Paris"
    test_path = "/mnt/l1/auto_top/SN3/data/Paris/images/AOI_3_Paris/seg_test_data"

    sn3_im_nums = [42, 49, 50, 99, 100, 132, 193, 219]


    # arbitrary_img_fpath = f"/mnt/l1/auto_top/SN3/data/mumbai_test.png"
    print_transform_and_ref(fname_from_fnum(sn3_im_nums[3]))
    img, road_mask, building_mask = get_img_masks_raw(fname_from_fnum(sn3_im_nums[3]), road_model, building_model)
    # img, road_mask, building_mask = get_img_masks_raw(arbitrary_img_fpath, road_model, building_model)
    overlay = blend_both_masks(img, road_mask, building_mask)
    ref_road = refine_road_mask(road_mask)
    ref_building = refine_building_mask(building_mask)
    overlay_refined = blend_both_masks(img, ref_road, ref_building)

    # plot_cv2(overlay_refined)
    # plot_before_after(ref_road, skeletonize_road_mask(ref_road), tag="skeletonized")
    building_polygon_mask = polygonize_building_mask(ref_building)
    road_skeleton_mask = skeletonize_road_mask(ref_road)
    burned_img = burn_outlines(overlay_refined, road_skeleton_mask, building_polygon_mask)
    # plot_cv2(burned_img)
    get_road_graph(road_skeleton_mask)
    # plot_before_after(ref_building, p_building_mask, tag='polygonized')
    