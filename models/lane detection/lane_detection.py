"""Detect lane boundaries in an image – contributed by Li Ma
"""

from . import config
from .camera import build_undistort_function
from .line_detection import LineDetector
from .transform import build_trapezoidal_bottom_roi_crop_function
from .transform import build_default_warp_transformer
from .utility import make_pipeline

import numpy as np
from skimage.morphology import remove_small_objects, remove_small_holes
import cv2

np.random.seed(1337)

class LaneDetector(object):
    def __init__(self):
        # setup the pipe to get the line image and meter-per-pixel measurements
        # 1. undistort function
        self.undistort = build_undistort_function()
        # 2. line detection function
        self.detect_line = LineDetector().detect
        # 3. roi crop function
        self.roi_crop = build_trapezoidal_bottom_roi_crop_function()
        # 4. bird-eye transformer object - it's stateful
        self.transformer = build_default_warp_transformer()
        # # 5. use the in-class meter-per-pixel on x an dy
        # self.transformer.x_mpp = 30/720
        # self.transformer.y_mpp = 3.7/700
    def detect_video(self, clip):
        """`clip`: A VideoFileClip from moviepy.editor
        Returns: An output VideoClip with marked lanes and estimated parameters
        """
        # store detection result from last frame, (lane_pixels, lane_params, models_in_pixel)
        self.last_detection = None
        return clip.fl_image(self.process_frame)
    def process_frame(self, frame):
        undistorted_img = self.undistort(frame)
        # lane_img = self.detect_line(self.transformer.transform(undistorted_img))
        image_pipe = make_pipeline([
                    self.detect_line,
                    self.roi_crop,
                    self.transformer.binary_transform
        ])
        lane_img = image_pipe(undistorted_img)
        H, W = lane_img.shape[:2]
        try:
            if self.last_detection is None: # first frame, search from scratch
                lane_params, marked_lane_img, lane_pixels, models_in_pixel = self.detect_image(frame, return_full_estimate=True)
                self.last_detection = (lane_pixels, lane_params, models_in_pixel)
                return marked_lane_img
            else: # try faster estimate using previous detection result
                # restore last result
                last_lane_pixels, last_lane_params, last_models_in_pixel = self.last_detection
                last_lmodel, _, last_rmodel = last_models_in_pixel
                # build new lane_pixels, lane_params, models_in_pixel
                window, stride = 50, 10
                search_ys = list(range(0, H-window+1, stride))
                search_lxs = np.polyval(last_lmodel, search_ys).astype(np.int32)
                search_rxs = np.polyval(last_rmodel, search_ys).astype(np.int32)
                lxs, mxs, rxs, lys, mys, rys = [], [], [], [], [], []
                for lx, rx, yy in zip(search_lxs, search_rxs, search_ys):
                    lregion = lane_img[yy:yy+stride, lx-window:lx+window]
                    ys, xs = np.where(lregion>0)
                    for y, x in zip(ys, xs):
                        lxs.append(x+lx-window)
                        lys.append(y+yy)
                    rregion = lane_img[yy:yy+stride, rx-window:rx+window]
                    ys, xs = np.where(rregion>0)
                    for y, x in zip(ys, xs):
                        rxs.append(x+rx-window)
                        rys.append(y+yy)
                ## this is probably not the best way of estimating the middle of lanes
                ## but it's worth minimizing the api
                left_lane, right_lane = dict(zip(lys, lxs)), dict(zip(rys, rxs))
                for y in range(H):
                    if y in left_lane and y in right_lane:
                        mys.append(y)
                        mxs.append( (left_lane[y]+right_lane[y])/2 )
                lxs,lys,mxs,mys,rxs,rys = map(np.array, [lxs,lys,mxs,mys,rxs,rys])
                lane_pixels = (lxs, lys, mxs, mys, rxs, rys)
                lane_params, _ = self.estimate_lane_params(lane_pixels, (W,H),
                    self.transformer.x_mpp, self.transformer.y_mpp)
                # smooth out lane parameters by an moving average
                lane_params = [p*0.9 + 0.1*p0 for p, p0 in zip(lane_params, last_lane_params)]
                _, models_in_pixel = self.estimate_lane_params(lane_pixels, (W, H), 1, 1)
                # update last result - if the detection result is not good enough (two lanes are not roughly in parallel)
                # force to detect from scracth at the next frame
                lcoeff, rcoeff = models_in_pixel[0][1], models_in_pixel[-1][1]
                if np.abs(lcoeff-rcoeff) >= 0.8*np.abs(max(lcoeff, rcoeff)):
                    self.last_detection = None
                    # print("detect from the scatch at next frame")
                else:
                    self.last_detection = (lane_pixels, lane_params, models_in_pixel)
        except:
            # if anything goes wrong, just use the previous result as default
            # or detect nothing if it is also the first frame
            if self.last_detection is None: return frame
            lane_pixels, lane_params, models_in_pixel = self.last_detection

        l_curvature, r_curvature, offset = lane_params
        text = "curvature: %.2f, %s of center: %.2f" % (r_curvature,
                                                        "left" if offset > 0 else "right",
                                                        np.abs(offset))
        marked_lane_img = self.draw_lanes(undistorted_img, models_in_pixel, text)
        return marked_lane_img


    def detect_image(self, img, return_full_estimate=False):
        """`img`: raw image from camera
        Returns:
            - `lane_params`: radius_of_curvature, offset from camera center
            - `marked_lane_img`: color img with lanes marked in different colors
        """
        # pipeline from raw to lane image
        undistorted_img = self.undistort(img)
        # lane_img = self.detect_line(self.transformer.transform(undistorted_img))
        image_pipe = make_pipeline([
                    self.detect_line,
                    self.roi_crop,
                    self.transformer.binary_transform
        ])
        lane_img = image_pipe(undistorted_img)
        lane_pixels = self.get_lane_pixels(lane_img)

        # lane_img = remove_small_holes(lane_img, min_size=128)
        # lane_img = remove_small_objects(lane_img, min_size=100)
        # return None, np.dstack([lane_img, lane_img, lane_img]).astype(np.uint8) * 255

        # get the real lane curvature and center offset
        H, W = lane_img.shape[:2]
        lane_params, _ = self.estimate_lane_params(lane_pixels, (W,H),
            self.transformer.x_mpp, self.transformer.y_mpp)
        # draw the lane back
        l_curvature, r_curvature, offset = lane_params
        # print(l_curvature, r_curvature)
        text = "curvature: %.2f, %s of center: %.2f" % (r_curvature,
                                                        "left" if offset > 0 else "right",
                                                        np.abs(offset))
        _, models_in_pixel = self.estimate_lane_params(lane_pixels, (W, H), 1, 1)
        marked_lane_img = self.draw_lanes(undistorted_img, models_in_pixel, text)
        if return_full_estimate:
            return lane_params, marked_lane_img, lane_pixels, models_in_pixel
        else:
            return lane_params, marked_lane_img


    def get_lane_pixels(self, lane_img):
        H, W = lane_img.shape[:2]
        lane_img = remove_small_holes(lane_img, min_size=128)
        lane_img = remove_small_objects(lane_img, min_size=128+16)
        window, stride = 50, 10
        lxs, lys = [], [] # lane left boundary coordinates
        mxs, mys = [], [] # lane middle coordinates
        rxs, rys = [], [] # lane right boundary coordinates
        for offset in range(0, H-window+1, stride):
            region = lane_img[offset:offset+window, :]
            ys, xs = np.where(region > 0)
            if len(xs) <= 25: continue
            xs = np.array(sorted(xs))
            i = np.argmax(np.diff(xs))
            left_xs = xs[:i-3]
            right_xs = xs[i+1:]
            is_valid_region = (W/3 <= (xs.max()-xs.min()) <= W*5/6)
            if is_valid_region:
                lx = np.mean(left_xs)#np.min(xs)#
                rx = np.median(right_xs)#np.max(xs)#
                mx = (lx + rx)/2

                my = np.mean(ys) + offset
                ly = my
                ry = my
                if True:#len(mxs) == 0 or np.abs(mx - np.mean(mxs)) <= 100:
                    if left_xs.max()-left_xs.min() <= W/10:
                        lxs.append(lx)
                        lys.append(ly)
                    if right_xs.max()-right_xs.min() <= W/10:
                        rxs.append(rx)
                        rys.append(ry)
                    if (left_xs.max()-left_xs.min() <= W/10) and (right_xs.max()- right_xs.min() <= W/10):
                        mxs.append(mx)
                        mys.append(my)

        lxs,lys,mxs,mys,rxs,rys = map(np.array, [lxs,lys,mxs,mys,rxs,rys])
        return lxs, lys, mxs, mys, rxs, rys

    def estimate_lane_params(self, lane_pixels, img_size,
        x_mpp = 30/720, y_mpp = 3.7/700):
        """`lane_img`: binary lane image
            `img_size`: width and height of image
            `x_mpp`: meter per pixel on x axis, default=1
            `y_mpp`: meter per pixel on y axis, default=1
        Returns:
            - `l_curvature`: radius of curvature for left lane, as ${(1+(2Ay+B)^2)^{3/2}}\over{|2A|}$
            - `r_curvature`: radius of curvature for right lane
            - `center offset`: as $lane_center_x - image_center_x$ in pixels
            - `lmodel`, `mmodel`, `rmodel`: 2nd polynomial model for left, center and right lane.
        """
        lxs,lys,mxs,mys,rxs,rys = lane_pixels
        W, H = img_size


        lmodel = np.polyfit(lys*y_mpp, lxs*x_mpp, 2)
        mmodel = np.polyfit(mys*y_mpp, mxs*x_mpp, 2)
        rmodel = np.polyfit(rys*y_mpp, rxs*x_mpp, 2)


        y = H*y_mpp

        A, B = lmodel[:2]
        l_curvature = (1 + (2*A*y+B)**2)**1.5 / np.abs(2*A)

        A, B = rmodel[:2]
        r_curvature = (1 + (2*A*y+B)**2)**1.5 / np.abs(2*A)

        offset = np.polyval(mmodel, y) - W*x_mpp/2
        return (l_curvature, r_curvature, offset), (lmodel, mmodel, rmodel)

    def draw_lanes(self, undistorted_img, models_in_pixel, text=""):
        H, W = undistorted_img.shape[:2]
        # _, models_in_pixel = self.estimate_lane_params(lane_pixels, (W, H), 1, 1)
        lmodel, mmodel, rmodel = models_in_pixel
        y_span = range(0, undistorted_img.shape[0]+1)
        lxhat = np.polyval(lmodel, y_span).astype(np.int32)
        mxhat = np.polyval(mmodel, y_span).astype(np.int32)
        rxhat = np.polyval(rmodel, y_span).astype(np.int32)


        lane_layer = np.zeros_like(undistorted_img, dtype=np.uint8)
        # lane_layer = cv2.fillPoly(lane_layer, [np.array([[lxhat[0], 0], [lxhat[-1], H],
        #                                         [rxhat[-1], H], [rxhat[0], 0]])], (0, 64, 0))

        lane_layer = cv2.fillPoly(lane_layer, [np.array(list(zip(lxhat, y_span))
                                                        + list(zip(rxhat, y_span))[::-1])], (0, 64, 0))
        for xs, col in zip([lxhat, mxhat, rxhat],
                           [(255,0,0), (255,0,0), (255,0,0)]):
            pts = np.array([(x, y) for x, y in zip(xs, y_span)])
            lane_layer = cv2.polylines(lane_layer, [pts], isClosed=False, color=col, thickness=20)



        lane_layer = self.transformer.transform(lane_layer, inverse=True)
        lane_img = cv2.addWeighted(undistorted_img, 1., lane_layer, 1., 1)
        lane_img = cv2.putText(lane_img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
            1.5, (255, 255, 0), 2)
        return lane_img
