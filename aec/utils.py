import cv2
import copy
import numpy as np
from grace.utils import *

class KeypointDetection(object):
    
    def __init__(self, crop_size=64):
        self.crop_size = crop_size
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        self.calib_params = load_json('config/calib/calib_params.json')
    
    def crop_image(self, image, x, y):
        # Calculate the coordinates for cropping
        img = copy.deepcopy(image)
        x_start = x - int(self.crop_size/2)
        y_start = y - int(self.crop_size/2)
        x_end = x_start + int(self.crop_size)
        y_end = y_start + int(self.crop_size)

        # Crop the image
        cropped_image = img[y_start:y_end, x_start:x_end]

        return cropped_image
    
    def assign_target(self, image, x, y):
        self.target_image = self.crop_image(image, x, y)
    
    def distance_from_target(self, new_image, eye):
        """After motor movement
        """
        self.new_image = new_image
        self.gray_target_image = cv2.cvtColor(self.target_image, cv2.COLOR_RGB2GRAY)
        self.gray_new_image = cv2.cvtColor(self.new_image, cv2.COLOR_RGB2GRAY)
        
        keypoints1, descriptors1 = self.sift.detectAndCompute(self.gray_target_image, None)
        keypoints2, descriptors2 = self.sift.detectAndCompute(self.gray_new_image, None)

        matches = self.matcher.match(descriptors1, descriptors2)
        sorted_matches = sorted(matches, key=lambda x: x.distance)

        # Select the top match
        top_match = sorted_matches[0]

        # Retrieve keypoint indices from the top match
        query_idx = top_match.queryIdx  # target
        train_idx = top_match.trainIdx

        # Retrieve corresponding keypoints
        keypoint1 = keypoints1[query_idx]  # target
        keypoint2 = keypoints2[train_idx]

        distance_to_cal_center_x = keypoint2.pt[0]-(keypoint1.pt[0]-self.crop_size/2)-self.calib_params[eye]['x_center']
        distance_to_cal_center_y = self.calib_params[eye]['y_center']-(self.crop_size/2-keypoint1.pt[1])-keypoint2.pt[1]

        magnitude = np.sqrt(distance_to_cal_center_x**2 + distance_to_cal_center_y**2)
        
        # Plotting
        self.disp_img = cv2.drawMatches(self.gray_target_image,keypoints1,self.gray_new_image,keypoints2,sorted_matches[:1],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        
        return magnitude, distance_to_cal_center_x, distance_to_cal_center_y