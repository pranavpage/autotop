# Script to test trained model
# !pip install git+https://github.com/qubvel/efficientnet
# !pip install git+https://github.com/qubvel/classification_models.git
# !pip install git+https://github.com/qubvel/segmentation_models
# !pip install -U git+https://github.com/albu/albumentations
import tensorflow as tf 
import cv2, os, re, sknw, rasterio
import rasterio.warp
from osgeo import gdal
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
    def __init__(self, aoi_path):
        self.im = []
        self.root_dir = aoi_path
        self.masks_path = f"{self.root_dir}/PS-RGB_8bit/raw_masks"
        self.proc_path = f"{self.root_dir}/PS-RGB_8bit/processed"
        self.road_kernel_size = 9
        self.building_kernel_size = 13
    def load_img_from_im_num(self, im_num):
        im_path = f"{self.root_dir}/PS-RGB_8bit/SN3_roads_train_AOI_3_Paris_PS-RGB_img{im_num}.tif"
        self.im_path = im_path
        im = imread(im_path)
        im = im[:,:,:3]

        # Pad the image to make both dimensions divisible by 32
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
        return self.im 
    def load_models(self):
        sn3_model_path = "/mnt/l1/auto_top/SN3/models/unet_sn3_high.h5"
        sn2_model_path = "/mnt/l1/auto_top/SN2/models/unet_sn2.h5"
        self.road_model = tf.keras.models.load_model(sn3_model_path, compile=False)
        self.building_model = tf.keras.models.load_model(sn2_model_path, compile=False)
        return self.road_model, self.building_model
    def get_raw_masks(self):
        road_mask_path = f"{self.masks_path}/road_mask_img{self.im_num}.png"
        w_p, h_p, _ = self.im.shape
        if(os.path.isfile(road_mask_path)):
            self.road_mask = cv2.imread(road_mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            road_mask_p = (self.road_model.predict(np.expand_dims(self.im, axis=0)/255.0)).reshape(w_p,h_p)
            road_mask_p*=255.0
            self.road_mask = road_mask_p.astype(np.uint8)
            cv2.imwrite(road_mask_path, self.road_mask)
        
        building_mask_path = f"{self.masks_path}/building_mask_img{self.im_num}.png"
        if(os.path.isfile(building_mask_path)):
            self.building_mask = cv2.imread(building_mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            building_mask_p = (self.building_model.predict(np.expand_dims(self.im, axis=0)/255.0)).reshape(w_p,h_p)
            building_mask_p*=255.0
            self.building_mask = building_mask_p.astype(np.uint8)
            cv2.imwrite(building_mask_path, self.building_mask)
        
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
        plt.show()
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

if(__name__=="__main__"):
    # sn3_model_path = "/mnt/l1/auto_top/SN3/models/unet_sn3_high.h5"
    # sn2_model_path = "/mnt/l1/auto_top/SN2/models/unet_sn2.h5"

    # road_model = tf.keras.models.load_model(sn3_model_path, compile=False)
    # building_model = tf.keras.models.load_model(sn2_model_path, compile=False)

    aoi_path = "/mnt/l1/auto_top/SN3/data/Paris/images/AOI_3_Paris"
    # test_path = "/mnt/l1/auto_top/SN3/data/Paris/images/AOI_3_Paris/seg_test_data"

    sn3_im_nums = [42, 49, 50, 99, 100, 132, 193, 219]
    # P1 TODO : change root, change image loading
    map1 = SegmentAndMap(aoi_path=aoi_path)
    map1.load_img_from_im_num(sn3_im_nums[3])
    map1.get_transform_information()
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

    # # arbitrary_img_fpath = f"/mnt/l1/auto_top/SN3/data/mumbai_test.png"
    # print_transform_and_ref(fname_from_fnum(sn3_im_nums[3]))
    # img, road_mask, building_mask = get_img_masks_raw(fname_from_fnum(sn3_im_nums[3]), road_model, building_model)
    # # img, road_mask, building_mask = get_img_masks_raw(arbitrary_img_fpath, road_model, building_model)
    # overlay = blend_both_masks(img, road_mask, building_mask)
    # ref_road = refine_road_mask(road_mask)
    # ref_building = refine_building_mask(building_mask)
    # overlay_refined = blend_both_masks(img, ref_road, ref_building)

    # # plot_cv2(overlay_refined)
    # # plot_before_after(ref_road, skeletonize_road_mask(ref_road), tag="skeletonized")
    # building_polygon_mask = polygonize_building_mask(ref_building)
    # road_skeleton_mask = skeletonize_road_mask(ref_road)
    # burned_img = burn_outlines(overlay_refined, road_skeleton_mask, building_polygon_mask)
    # # plot_cv2(burned_img)
    # get_road_graph(road_skeleton_mask)
    # # plot_before_after(ref_building, p_building_mask, tag='polygonized')
    