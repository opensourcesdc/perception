#Contributed by Li Ma, github:dolaameng

from . import config
from . import classification

from scipy.misc import imresize
import numpy as np

from skimage.measure import label, regionprops
from skimage.morphology import closing, square, opening
import cv2

class VehicleDetector(object):
    def __init__(self):
        """Main control parameters for balance between recall vs precision:
        - stride
        - scale
        """
        # vehicle classification model
        print("building vehicle classification model")
        self.vehicle_model = classification.build_model()
        # parameters for sliding window
        self.window_size = (64, 64)
        self.rows = (0.5, 1)
        self.cols = (0.35, 1)
        self.stride = (32, 32)
        # parameters for layer pyramid
        self.scale = 1.2
        # heatmap threshold
        self.heatmap_thr = 2.5
        # for video detection
        self.last_heatmap = None

    def slide_window(self, image, 
        rows=(0, 1), cols=(0, 1), 
        window_size=(64, 64), stride=(8, 8), scale=1):
        """Slide a window through an image and return all the patches.
        `image`: image to slide through
        `rows`: tuple of (start_row, end_row)
        `cols`: tuple of (start_col, end_col)
        `window_size`: (nrow, ncol)
        `stride`: (row_stride, col_stride)
        `scale`: scale of pyramid layer from original. used to scale stride
        accordingly.
        """
        wdow_nr, wdow_nc = window_size
        img_nr, img_nc = image.shape[:2] 

        start_row, end_row = int(img_nr*rows[0]), int(img_nr*rows[1])
        start_col, end_col = int(img_nc*cols[0]), int(img_nc*cols[1])
        stride_row, stride_col = stride
        stride_row = max(1, int(stride_row / scale))
        stride_col = max(1, int(stride_col / scale))
        # print (stride_row, stride_col, scale)
        
        for r in range(start_row, end_row+1, stride_row):
            for c in range(start_col, end_col+1, stride_col):
                if r+wdow_nr > img_nr:
                    break
                if c+wdow_nc > img_nc: continue
                patch = image[r:r+wdow_nr, c:c+wdow_nc]
                bbox = [(c, r), (c+wdow_nc, r+wdow_nr)]
                yield (patch, bbox)
    def get_image_pyramid(self, image, scale=1.25, min_rc=(256, 256)):
        """Get a pyramid of layers from original images by downscaling at each step 
        at a fixed scale factor. 
        It serves as a generator that returns layer_image and scale_factor at each step.
        `scale`: downscale rate for each layer
        `min_rc`: minimum # of rows and columns at the very top of pyramid.
        """
        min_r, min_c = min_rc
        multiple = 1
        yield image, multiple
        while True:
            multiple *= scale
            img = imresize(image, 1/multiple, )
            if img.shape[0] < min_r or img.shape[1] < min_c:
                break
            yield img, multiple
    def get_pyramid_slide_window(self, image, pyramid_params=None, window_params=None):
        """Combine the generator of pyramid of layers and sliding window on each layer, 
        It returns a new generator that yields an image patch, its bounding box in original 
        image and its scaling factor from the original.
        """
        pyramid_params = pyramid_params or {}
        window_params = window_params or {}
        for layer, multiple in self.get_image_pyramid(image, **pyramid_params):
            window_params.update({"scale": multiple})
            for patch, [(w0, h0), (w1, h1)] in self.slide_window(layer, **window_params):
                bbox_in_original = [(int(w0*multiple), int(h0*multiple)), 
                            (int(w1*multiple), int(h1*multiple))]
                yield (patch, bbox_in_original, multiple)
    def preprocess(self, image):
        """Preprocess Image, e.g., resizing
        """
        return image
    def get_heatmap(self, image):
        """Merge boxes by creating a heatmap.
        """
        image = self.preprocess(image)
        pyramid_params = {"scale": self.scale}
        window_params = {"rows": self.rows, "cols": self.cols, "window_size": self.window_size, "stride": self.stride}
        patches, bboxes, scales = zip(*list(self.get_pyramid_slide_window(image, pyramid_params, window_params)))
        # patches, bboxes = self.get_random_patches(image)
        is_vehicle = (self.vehicle_model.predict(np.array(patches)) == "vehicle")
        vehicle_bboxes = np.array(bboxes)[is_vehicle]
        heatmap = np.zeros(image.shape[:2])
        for bbox in vehicle_bboxes:
            (c1, r1), (c2, r2) = bbox
            heatmap[r1:r2,c1:c2] += 1
        
        return heatmap

    def draw_merged_boxes(self, image, heatmap, return_bboxes=False):
        heatmap = heatmap >= self.heatmap_thr#2.5#4#
        # zeros = np.zeros_like(heatmap)
        # return (image*0.1 + 0.9*image*np.dstack([zeros, heatmap, zeros])).astype(np.uint8)

        # heatmap = closing(heatmap, square(3))
        heatmap = opening(heatmap, square(11))
        label_img = label(heatmap)
        bboxes = []
        plot_img = image.copy()
        regions = regionprops(label_img)
        avg_area = 1500#4500#np.median([r.area for r in regions])
        for region in regionprops(label_img):
            if region.area < avg_area: continue
            r1,c1,r2,c2 = region.bbox
            bboxes.append([(c1, r1), (c2, r2)])
            cv2.rectangle(plot_img, (c1, r1), (c2, r2), (255, 0, 0), 3)
        if return_bboxes:
            return plot_img, bboxes
        else:
            return plot_img

    def get_random_patches(self, image, n=2000):
        """Cheaper way of sampling patches, compared to sliding windown and image pyramid
        """
        R, C = image.shape[:2]
        minsize, maxsize = 64, 96
        r1s = np.random.randint(R//2, R-minsize, n)
        c1s = np.random.randint(int(C*0.45), C-maxsize, n)
        szs = np.random.randint(minsize, maxsize, n)
        r2s = np.minimum(r1s + szs, R)
        c2s = np.minimum(c1s + szs, C)
        bboxes = [ [(c1, r1), (c2, r2)] for r1, r2, c1, c2 in zip(r1s, r2s, c1s, c2s)]
        patches = [imresize(image[r1:r2, c1:c2, :], (64,64)) for r1, r2, c1, c2 in zip(r1s, r2s, c1s, c2s)]
        return patches, bboxes

    def detect_in_image(self, image):
        """Detect vehicles in an image.
        It returns a list of bboxes for each vehicle in the original image.
        """
        
        # return vehicle_bboxes
        heatmap = self.get_heatmap(image)
        detection_img = self.draw_merged_boxes(image, heatmap)
        return detection_img

    def detect_in_video(self, video):
        """Detect vehicles in `moviepy.editor.VideoFileClip`
        """
        self.last_heatmap = None
        def process_frame(frame):
            if self.last_heatmap is None:
                self.last_heatmap = self.get_heatmap(frame)
                detection_img = self.draw_merged_boxes(frame, self.last_heatmap)
            else:
                heatmap = self.get_heatmap(frame)
                combined_heatmap = heatmap * 0.8 + self.last_heatmap * 0.2
                detection_img = self.draw_merged_boxes(frame, combined_heatmap)
                self.last_heatmap = combined_heatmap
            return detection_img
        processed_video = video.fl_image(process_frame)
        return processed_video
        