"""
Vehicle classification models and features.
Contributed by Li Ma, github:dolaameng
"""

from . import config
from os import path
import pickle

import numpy as np
np.random.seed(1337)

from skimage import io, feature, color
from glob import glob
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.image import extract_patches_2d

from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.misc import imresize

class ImageFeatExtractor(BaseEstimator):
    """Feature extractor for vehicle/non-vehicle images.
    Two types of features used: 
        1. hog of gray 
        2. color histogram (to reduce false positive?)
    """
    def __init__(self, pixels_per_cell=(8,8), 
        cells_per_block=(2,2), hist_nbins=32, **kwargs):
        # for hog features
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        # for color histogram features
        self.hist_nbins = hist_nbins
        # feature extractors
        self.extract_hog = lambda img: feature.hog(color.rgb2gray(img), 
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block)
        self.extract_hist = lambda img, chann: np.histogram(img[:,:,chann],
                bins=self.hist_nbins, range=(0, 256))[0].astype(np.float)
        # feature standarizer
        self.hog_ss = StandardScaler()
        self.rhist_ss = StandardScaler()
        self.ghist_ss = StandardScaler()
        self.bhist_ss = StandardScaler()
    def fit_transform(self, images, labels=None):
        hog_feats = np.vstack([self.extract_hog(im) for im in images])  
        hog_feats = self.hog_ss.fit_transform(hog_feats)

        # return np.hstack([hog_feats])

        rhist_feats = np.vstack([self.extract_hist(im, 0) for im in images])
        rhist_feats = self.rhist_ss.fit_transform(rhist_feats)

        ghist_feats = np.vstack([self.extract_hist(im, 1) for im in images])
        ghist_feats = self.ghist_ss.fit_transform(ghist_feats)

        bhist_feats = np.vstack([self.extract_hist(im, 2) for im in images])
        bhist_feats = self.bhist_ss.fit_transform(bhist_feats)


        return np.hstack([hog_feats, rhist_feats, ghist_feats, bhist_feats])

    def fit(self, images, labels=None):
        self.fit_transform(images)
        return self
    def transform(self, images, labels=None):
        hog_feats = np.vstack([self.extract_hog(im) for im in images])
        hog_feats = self.hog_ss.transform(hog_feats)

        # return np.hstack([hog_feats])

        rhist_feats = np.vstack([self.extract_hist(im, 0) for im in images])
        rhist_feats = self.rhist_ss.transform(rhist_feats)

        ghist_feats = np.vstack([self.extract_hist(im, 1) for im in images])
        ghist_feats = self.ghist_ss.transform(ghist_feats)

        bhist_feats = np.vstack([self.extract_hist(im, 2) for im in images])
        bhist_feats = self.bhist_ss.transform(bhist_feats)


        return np.hstack([hog_feats, rhist_feats, ghist_feats, bhist_feats])

class VehicleClassifier(BaseEstimator, ClassifierMixin):
    """Vehicle detection model based on HOG feature and 
    Linear SVC model.
    """
    def __init__(self, **kwargs):
        # parameter for LinearSVC
        self.C = kwargs.get("C", 1)
        self.model = Pipeline([
            ("feature_extractor", ImageFeatExtractor(**kwargs)),
            ("svc", LinearSVC(C=self.C))
            # ("svc", SVC(C=self.C, kernel="linear", probability=True))
        ])
    def fit(self, images, labels):
        """`images`: list of images of fixed shape
        `labels`: list of labels, with the same len of images
        Return: a trained vehicle detection model
        """
        
        self.model.fit(images, labels)
        return self
    def best_fit(self, images, labels):
        """Find best models by random search
        """
        params = {
            'svc__C': np.logspace(-3, 3, 10),
            'feature_extractor__pixels_per_cell': [(6, 6), (8, 8)],
            'feature_extractor__cells_per_block': [(2, 2), (3, 3)],
            'feature_extractor__hist_nbins':[32, 64]
        }
        searcher = RandomizedSearchCV(self.model, 
            params, 
            cv=3, n_jobs=1, verbose=2, n_iter=15)
        searcher.fit(images, labels)
        self.model = searcher.best_estimator_
        self.best_params = searcher.best_params_
        return self
    def predict(self, images):
        return self.model.predict(images)
    def score(self, images, labels):
        return self.model.score(images, labels)


def load_data():
    vehicle_files = glob(config.vehicle_image_files)
    nonvehicle_files = glob(config.nonvehicle_image_files)
    print("loaded %i vehicle images and %i nonvehicle images" % 
        (len(vehicle_files), len(nonvehicle_files)))
    vehicle_imgs = [io.imread(f) for f in vehicle_files]
    nonvehicle_imgs = [io.imread(f) for f in nonvehicle_files]
    images = vehicle_imgs + nonvehicle_imgs
    labels = ["vehicle"]*len(vehicle_imgs)+["nonvehicle"]*len(nonvehicle_imgs)
    return images, labels

def enhance_negative_data():
    """Enhance the dataset by generating more negative samples
    from test images - to reduce the false positive rate
    """
    whole_images = []
    for f in glob("./test_images/*.jpg"):
        # whole_image = io.imread("../test_images/test2.jpg")
        whole_image = io.imread(f)
        R, C = whole_image.shape[:2]
        if "test2.jpg" in f:
            whole_image = whole_image[R//2:, :, :]
        elif "test5.jpg" in f:
            whole_image = whole_image[500:, :, :]
        else:
            whole_image = whole_image[R//2:, :C//2, :]
        whole_images.append(whole_image)

    images = []
    for whole_image in whole_images:
        for scale in range(3):
            patch_size = int(64*1.2**scale)
            patches = extract_patches_2d(whole_image, (patch_size, patch_size), max_patches=3000)
            patches = [imresize(p, (64, 64)) for p in patches]
            images += patches
    labels = ["nonvehicle"] * len(images)
    print("ehanced the dataset with %i negative images" % len(labels))
    return images, labels

def enhance_positive_data(car_images):
    """Enhance the dataset by generating more positive images
    through random shifting and rotating of car images.
    """
    import scipy.ndimage as ndi
    def transform_matrix_offset_center(matrix, x, y):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix

    def apply_transform(x, transform_matrix, channel_index=2, fill_mode='nearest', cval=0.):
        x = np.rollaxis(x, channel_index, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]
        channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                          final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_index+1)
        return x
    def random_rotation(img, degree=30):
        theta = np.pi / 180 * np.random.uniform(-degree, degree)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        h, w = img.shape[0], img.shape[1]
        transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
        x = apply_transform(img, transform_matrix, channel_index=2)
        return x

    def random_shift(img, shiftw=0.15, shifth=0.15):
        h, w = img.shape[0], img.shape[1]
        tx = np.random.uniform(-shifth, shifth) * h
        ty = np.random.uniform(-shiftw, shiftw) * w
        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        transform_matrix = translation_matrix  # no need to do offset
        x = apply_transform(img, transform_matrix, 2)
        return x

    def random_crop(img):
        R, C = img.shape[:2]
        r0 = np.random.randint(0, R*2//5)
        r1 = np.random.randint(R*3//5, R)
        c0 = np.random.randint(0, C*2//5)
        c1 = np.random.randint(C*3//5, C)
        return imresize(img[r0:r1, c0:c1], (R, C))

    images = []
    for car_image in car_images:
        rotated = [random_rotation(car_image, 5) for _ in range(3)]
        shifted = [random_shift(car_image, 0.05, 0.05) for _ in range(2)]
        cropped = [random_crop(car_image) for _ in range(5)]
        images += rotated
        images += shifted
    labels = ["vehicle"] * len(images)
    print("enhanced the dataset with %i positive samples" % len(labels))
    return images, labels

    

def fit_best_model():
    images, labels = load_data()
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2)
    classifier = VehicleClassifier()
    classifier.best_fit(train_images, train_labels)
    print("best model performance on test:", classifier.score(test_images, test_labels))
    return classifier

def build_model():
    if path.exists(config.model_file):
        classifier = pickle.load(open(config.model_file, "r"))
        return classifier
    images, labels = load_data()
    enhanced_negative_images, enhanced_negative_labels = enhance_negative_data()
    nvhicles = (np.array(labels)=="vehicle").sum()
    enhanced_pos_images, enhanced_pos_labels = enhance_positive_data(images[:nvhicles])
    images += enhanced_negative_images
    images += enhanced_pos_images
    labels += enhanced_negative_labels
    labels += enhanced_pos_labels
    images, labels = shuffle(images, labels)
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.1)
    classifier = VehicleClassifier(
            pixels_per_cell=(8,8),
            cells_per_block=(2,2),
            hist_nbins=16,
            C=5e-1)
    classifier.fit(train_images, train_labels)
    print("built model performance on test:", classifier.score(test_images, test_labels))
    # classifier.fit(images, labels)
    # pickle.dump(classifier, open(config.model_file, "w"))
    return classifier