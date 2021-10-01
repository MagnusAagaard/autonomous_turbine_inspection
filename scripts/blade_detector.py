#!/usr/bin/env python3
import cv2
from pylsd.lsd import lsd
import numpy as np

from math import log, pow
import random

class BladeDetector:
    def __init__(self, img_path=None, save_result=False):
        # Initialisation
        self.save_result = save_result
        if self.save_result:
            self.save_path = './output/' + img_path.split('/')[-1]
        self.img = cv2.imread(img_path)
        #self.img = cv2.resize(self.img, (1920, 1080))
        self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def get_mask(self, file_annot):
        """
        Return the mask based on CIE-LAB color segmentation.
        The mask is used for filtering lines in detect()

        @type   file_annot: path
        @param  file_annot: path to annotated image file
        @rtype:   list
        @return:  the CIE-LAB mask as a list of length 2: (mean, std)
        """ 
        # Calculate a CIE-LAB color segmentation mask based on an annotated image (red pixel values)
        img_annot = cv2.imread(file_annot)
        lower_limit = (0, 0, 245)
        upper_limit = (10, 10, 256)
        mask = cv2.inRange(img_annot,lower_limit, upper_limit)
        img_lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        mean_lab, std_lab = cv2.meanStdDev(img_lab, mask = mask)
        self.mask_lab = ((mean_lab-2*std_lab).flatten(), (mean_lab+2*std_lab).flatten())
        print("Mean: {}, std: {}".format(mean_lab, std_lab))
        print("Acutal mask range: {}".format(self.mask_lab))

    def detect(self):
        """
        Run the blade detection algorithm. Non returning?
        """ 
        # Detect lines
        self.lines = lsd(self.gray_img)
        # Draw lines
        tmp_img = self.img.copy()
        for i in range(self.lines.shape[0]):
            pt1 = (int(self.lines[i, 0]), int(self.lines[i, 1]))
            pt2 = (int(self.lines[i, 2]), int(self.lines[i, 3]))
            width = self.lines[i, 4]
            cv2.line(tmp_img, pt1, pt2, (0,255,0), int(np.ceil(width / 2)))
        if self.save_result:
            cv2.imwrite(self.save_path[:-4] + '_no_filter.jpg', tmp_img)
        # Filter lines using the CIE-LAB mask
        img_lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        masked_image = cv2.inRange(img_lab, self.mask_lab[0], self.mask_lab[1])
        # Dilate the masked image, such that we get lines that are on the edge of the wind turbine
        kernel = np.ones((9,9), np.uint8)
        masked_image = cv2.dilate(masked_image, kernel)
        # Detect contours and only use large contours
        masked_image_zero = np.zeros_like(masked_image)
        contours, hierarchy = cv2.findContours(image=masked_image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        contours_to_keep = []
        for con in contours:
            if cv2.contourArea(con) > 50000:
                contours_to_keep.append(con)
        # Fill mask with large contours
        cv2.drawContours(masked_image_zero, contours_to_keep, -1, 255, -1)
        # Use the new mask
        masked_image = masked_image_zero
        if self.save_result:
            cv2.imwrite(self.save_path[:-4] + '_contour_mask.jpg', masked_image)
        # Filter using the new mask
        self.filtered_lines = []
        for i in range(self.lines.shape[0]):
            pt1 = (int(self.lines[i, 0]), int(self.lines[i, 1]))
            pt2 = (int(self.lines[i, 2]), int(self.lines[i, 3]))
            width = self.lines[i, 4]
            if masked_image[pt1[1], pt1[0]] != 0 and masked_image[pt2[1], pt2[0]] != 0:
                cv2.line(self.img, pt1, pt2, (0,255,0), int(np.ceil(width / 2)))
                self.filtered_lines.append(self.lines[i])
        print("Number of unfiltered lines: {}".format(len(self.lines)))
        print("Number of filtered lines: {}".format(len(self.filtered_lines)))
        if self.save_result:
            cv2.imwrite(self.save_path[:-4] + '_filtered.jpg', self.img)
        self.estimate_best_fit()
        if self.save_result:
            cv2.imwrite(self.save_path[:-4] + '_intersections.jpg', self.img)

    def estimate_best_fit(self):
        """
        Return the best model fit estimate using RANSAC on filtered lines.
        The turbine model parameters are: angle between lines, lengths,
        mean lengths, intersections and centroid.

        @type   lines: numpy array of shape (3,4)
        @param  lines: Each row is 4 numbers, y1, x1, y2, x2
        @rtype:   numpy array of shape (,5)
        @return:  the turbine model parameters
        """ 
        # Use RANSAC to estimate the best fit from the filtered lines
        n = 3   # Minimum number of parameters to estimate model
        p = 0.99    # Desired prob for finding a fit
        w = 0.5     # prob of choosing an inlier: w ~= inliers/(inliers+outliers)
        k = int(log(1-p) / log(1-pow(w,n))) + 1
        best_fit = None
        best_err = np.inf
        for i in range(k):
            maybe_inliers = np.array(random.sample(self.filtered_lines, n))
            maybe_model = self.get_turbine_model(maybe_inliers[:,:4])

    def get_turbine_model(self, lines):
        """
        Return the turbine model parameters given three lines.
        The turbine model parameters are: angle between lines, lengths,
        mean lengths, intersections and centroid.

        @type   lines: numpy array of shape (3,4)
        @param  lines: Each row is 4 numbers, y1, x1, y2, x2
        @rtype:   numpy array of shape (,5)
        @return:  the turbine model parameters
        """ 
        lines_homogenous = []
        intersections = []
        for l in lines:
            # Transform each point (y,x) to homogenous coordinates (x,y,1)
            p1 = np.flip(np.append(1,l[:2]))
            p2 = np.flip(np.append(1,l[2:]))
            # Calculate lines in homogenous coordinates
            lines_homogenous.append(np.cross(p1,p2))
        lines_homogenous = np.array(lines_homogenous)

        # Calculate intersections
        intersections = []
        intersections.append(np.cross(lines_homogenous[0], lines_homogenous[1]))
        intersections.append(np.cross(lines_homogenous[1], lines_homogenous[2]))
        intersections.append(np.cross(lines_homogenous[0], lines_homogenous[2]))
        intersections = np.array(intersections)
        # Back to cartesian space (x,y)
        intersections = np.array([intersections[i,:2] / intersections[i,2] for i in range(len(intersections))])
        img_pts = np.fliplr(intersections).astype(int)
        # Draw blue dots on intersection points
        for pt in img_pts:
            cv2.circle(self.img, tuple(pt), 3, (255,0,0), thickness=3)
        # Calculate centroid of intersections
        centroid = np.sum(intersections, axis=0)/len(intersections)
        # Red dots on centroids
        cv2.circle(self.img, tuple(np.flip(centroid).astype(int)), 3, (0,0,255),thickness=3)
        # Extend lines from centroid to edge of mask
        # Calculate length of each line,, mean length, angle between lines and we have what we need!

        return centroid


def main():
    bd = BladeDetector(img_path='./image_data/offshore_wind_turbine.jpg', save_result=True)
    bd.get_mask('./image_data/annotated_wind_turbine.jpg')
    bd.detect()

if __name__ == "__main__":
    main()