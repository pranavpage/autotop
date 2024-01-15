# Script to deploy models and merge predictions
import tensorflow as tf 
import cv2, os, re, sknw, rasterio
import rasterio.warp
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
os.environ["SM_FRAMEWORK"] = "tf.keras"
import matplotlib.pyplot as plt 
import numpy as np 
from efficientnet.tfkeras import EfficientNetB0
from keras.optimizers import Adam
from skimage.io import imread
from skimage.morphology import skeletonize
from segmentation_models.losses import bce_jaccard_loss, bce_dice_loss
from segmentation_models.metrics import iou_score

class SegmentAndMap:
    def __init__(self, data_path, models_path, proc_path, model_roads, model_buildings):
        self.im = []
        self.data_path = data_path
        self.models_path = models_path
        self.proc_path = proc_path
        self.model_roads = model_roads
        self.model_buildings = model_buildings
        self.road_kernel_size = 9
        self.building_kernel_size = 13
    def load_img_from_im_num(self, im_num, dset = 'road'):
        im_path = f"{self.data_path}/{dset}/img/img{im_num}.tif"
        self.im_path = im_path
        im = imread(im_path)
        im = im[:,:,:3]

        pad_height = (32 - im.shape[0] % 32) % 32
        pad_width = (32 - im.shape[1] % 32) % 32

        padded_img = cv2.copyMakeBorder(im, pad_height // 2, pad_height - pad_height // 2,
                                        pad_width // 2, pad_width - pad_width // 2,
                                        cv2.BORDER_REFLECT)
        w_p, h_p, _ = padded_img.shape
        print(f"Loaded {im_path}, resized {im.shape} to {padded_img.shape}")
        self.im = padded_img
        self.im_display = self.im
        self.im_num = im_num
        # P2 TODO : Load only once
        # Some img info
        with rasterio.open(self.im_path) as src:
            transform = src.transform
            bounds = src.bounds
            target_crs = 'EPSG:3857'
            target_width = src.width
            target_height = src.height

            dst_crs = target_crs
            dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds, dst_width=target_width, dst_height=target_height
            )

            pixel_size_x = abs(dst_transform[0])
            pixel_size_y = abs(dst_transform[4])


        self.im_bounds = bounds
        self.transform = transform
        self.pixel_sizes = [pixel_size_x, pixel_size_y]
        return self.im 
    def load_models(self):
        model_roads_path = f"{self.models_path}/{self.model_roads}"
        model_buildings_path = f"{self.models_path}/{self.model_buildings}" 
        self.road_model = tf.keras.models.load_model(model_roads_path, compile=False)
        self.building_model = tf.keras.models.load_model(model_buildings_path, compile=False)
        return self.road_model, self.building_model
    def get_raw_masks(self):
        w_p, h_p, _ = self.im.shape
        road_mask_p = (self.road_model.predict(np.expand_dims(self.im, axis=0)/255.0)).reshape(w_p,h_p)
        road_mask_p*=255.0
        self.road_mask = road_mask_p.astype(np.uint8)
        
        building_mask_p = (self.building_model.predict(np.expand_dims(self.im, axis=0)/255.0)).reshape(w_p,h_p)
        building_mask_p*=255.0
        self.building_mask = building_mask_p.astype(np.uint8)
        
        return self.road_mask, self.building_mask
    def blend_img_with_masks(self, alpha = 0.2):
        # Create an empty image for overlaying the masks
        overlay = cv2.cvtColor(self.im, cv2.COLOR_BGR2GRAY)
        overlay = cv2.merge([overlay, overlay, overlay])
        # Create red mask for buildings
        building_mask = cv2.merge([np.zeros_like(self.building_mask), np.zeros_like(self.building_mask), self.building_mask])

        alpha_building = alpha  # Adjust transparency (alpha) for buildings
        cv2.addWeighted(building_mask, alpha_building, overlay, 1 - alpha_building, 0, overlay)

        # Create green mask for roads with transparency (adjust alpha)
        road_mask = cv2.merge([np.zeros_like(self.road_mask), self.road_mask, np.zeros_like(self.road_mask)])
        alpha_road = alpha  # Adjust transparency (alpha) for roads
        cv2.addWeighted(road_mask, alpha_road, overlay, 1 - alpha_road, 0, overlay)
        self.im_display = overlay
        self.overlay = overlay
        return self.im_display
    
    def refine_masks(self, binarize_thr = 0.5, iters = 1, kernel_size = 5):
        # TODO : better to derive a road segmenter and a building segmenter from a class 
        _, b_road_mask = cv2.threshold(self.road_mask, int(binarize_thr*255), 255, cv2.THRESH_BINARY)
        final_road_mask = b_road_mask
        _, b_building_mask = cv2.threshold(self.building_mask, int(binarize_thr*255), 255, cv2.THRESH_BINARY)
        final_building_mask = b_building_mask
       
        road_kernel = np.ones((self.road_kernel_size, self.road_kernel_size), np.uint8) 
        final_road_mask = cv2.morphologyEx(b_road_mask, cv2.MORPH_CLOSE, road_kernel, iterations=iters)
        
        building_kernel = np.ones((self.building_kernel_size, self.building_kernel_size), np.uint8) 
        final_building_mask = cv2.morphologyEx(b_building_mask, cv2.MORPH_OPEN, building_kernel, iterations=iters)
        
        self.road_mask = final_road_mask
        self.building_mask = final_building_mask
        return self.road_mask, self.building_mask
    def skeletonize_road_mask(self):
        self.road_skeleton = np.uint8(skeletonize(self.road_mask)*255)
        _, b_road_skeleton = cv2.threshold(self.road_skeleton, 127, 255, cv2.THRESH_BINARY) 
        self.road_graph = sknw.build_sknw(b_road_skeleton)
        return self.road_skeleton, self.road_graph
    def polygonize_building_mask(self):
        contours, _ = cv2.findContours(self.building_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_polygons_image = np.zeros_like(self.building_mask, dtype=np.uint8)

        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, epsilon, True)

            # Draw the bounding polygon on the image
            cv2.polylines(bounding_polygons_image, [polygon], isClosed=True, color=255, thickness=1)

        print(f"Number of buildings = {len(contours)}")
        self.building_polygons = bounding_polygons_image
        return self.building_polygons
    def burn_outlines(self, dilate_kernel_size = 0):
        road_skeleton_green = cv2.merge([np.zeros_like(self.road_skeleton), self.road_skeleton, np.zeros_like(self.road_skeleton)])
        building_polygons_red = cv2.merge([np.zeros_like(self.building_polygons), np.zeros_like(self.building_polygons), self.building_polygons])
        kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
        if(dilate_kernel_size):
            road_skeleton_green = cv2.dilate(road_skeleton_green, kernel, iterations=1)
            building_polygons_red = cv2.dilate(building_polygons_red, kernel, iterations=1)
        ret_img = cv2.add(self.overlay, road_skeleton_green)
        ret_img = cv2.add(ret_img, building_polygons_red)
        self.im_display = ret_img
        return self.im_display
    def display_plt(self, save = False):
        # cv2.imshow('Image', self.im_display)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        plt_img = cv2.cvtColor(self.im_display, cv2.COLOR_BGR2RGB)
        w_p, h_p, _ = self.im_display.shape
        fig, ax = plt.subplots(facecolor='black', figsize=(12, 12))
        ax.imshow(plt_img)
        xlabels = [floatdeg2str(e) for e in np.linspace(self.im_bounds.left, self.im_bounds.right, 5)]
        ylabels = [floatdeg2str(e) for e in np.linspace(self.im_bounds.bottom, self.im_bounds.top, 5)]
        
        if(self.road_graph):
            nodes = self.road_graph.nodes()
            ps = np.array([nodes[i]['o'] for i in nodes])
            ax.plot(ps[:,1], ps[:,0], 'y.')
        ax.set_xticks(np.linspace(0, w_p, 5))
        ax.set_yticks(np.linspace(0, h_p, 5))

        ax.set_xticklabels(xlabels, color='white')
        ax.set_yticklabels(ylabels, color='white')

        ax.grid(color='w', dashes = (1, 5))
        num_pixels = int(1e2/self.pixel_sizes[0])
        scalebar = AnchoredSizeBar(ax.transData,
                           num_pixels, '100 m', 'lower right', 
                           pad=0.1,
                           color='white',
                           frameon=False,
                           size_vertical=1)
        ax.add_artist(scalebar)
        if save:
            fig.savefig(f"{self.proc_path}/img{self.im_num}.png")
        # plt.show()
        return 
    def display_cv2(self):
        cv2.imshow('Image', self.im_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 
    def get_transform_information(self):
        with rasterio.open(self.im_path) as src:
            transform = src.transform
            bounds = src.bounds
            target_crs = 'EPSG:3857'
            target_width = src.width
            target_height = src.height

            dst_crs = target_crs
            dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds, dst_width=target_width, dst_height=target_height
            )

            pixel_size_x = abs(dst_transform[0])
            pixel_size_y = abs(dst_transform[4])

            print(f'Pixel size (width) after reprojection: {pixel_size_x} meters')
            print(f'Pixel size (height) after reprojection: {pixel_size_y} meters')

        print("Transform Information:")
        print("Affine Matrix:")
        print(transform)
        self.im_bounds = bounds
        self.transform = transform
        self.pixel_sizes = [pixel_size_x, pixel_size_y]
        print(f"Bounds = {bounds}")
        print("\nNumber of Bands:", src.count)
        print("Raster Size (width, height):", src.width, src.height)
        print("CRS (Coordinate Reference System):", src.crs)
        return 

def floatdeg2str(fdeg):
    sign = -1 if fdeg < 0 else 1
    fdeg = abs(fdeg)
    sdeg = int(fdeg)
    smin = int((fdeg - sdeg)*60)
    ssec = (fdeg - sdeg - smin/60)*60*60
    return f"{sign * sdeg}° {smin}' {ssec:.2f}\""

def test_new_pipeline():
    print("Testing flow from data/ and to plots")
    curr_path = os.getcwd()
    print(f"Currrent path = {curr_path}")
    data_path = f"{curr_path}/data"
    models_path = f"{curr_path}/models"
    proc_path = f"{curr_path}/plots"
    model_roads = "unet_roads.h5"
    model_buildings = "unet_buildings.h5"
    road_im_nums = [288, 220, 434, 406, 75] # Some dense images
    map1 = SegmentAndMap(data_path=data_path, models_path = models_path, 
                         proc_path=proc_path, 
                         model_roads=model_roads, model_buildings=model_buildings)
    
    map1.load_img_from_im_num(road_im_nums[4])
    map1.load_models()
    map1.get_raw_masks()
    map1.refine_masks()
    map1.blend_img_with_masks()
    # map1.display()

    map1.skeletonize_road_mask()
    map1.polygonize_building_mask()
    map1.blend_img_with_masks()
    map1.burn_outlines(dilate_kernel_size=1)
    # map1.display_cv2()
    map1.display_plt(save=True)


if(__name__=="__main__"):
    # sn3_model_path = "/mnt/l1/auto_top/SN3/models/unet_sn3_high.h5"
    # sn2_model_path = "/mnt/l1/auto_top/SN2/models/unet_sn2.h5"

    # road_model = tf.keras.models.load_model(sn3_model_path, compile=False)
    # building_model = tf.keras.models.load_model(sn2_model_path, compile=False)
    test_new_pipeline()

    # P1 TODO : change root, change image loading
    # P1 TODO : draw proper lat long values 
    # P1 TODO : doesn't matter if mass prediction not being done, just do one prediction


    