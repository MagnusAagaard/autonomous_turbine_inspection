#!/usr/bin/env python3
import cv2
from pylsd.lsd import lsd
import numpy as np

from math import log, pow, sqrt
import random

class BladeDetector:
    def __init__(self, img_path=None, save_result=False):
        # Initialisation
        self.save_result = save_result
        if self.save_result:
            self.save_path = './scripts/output/' + img_path.split('/')[-1]
        self.img = cv2.imread(img_path)
        self.img_final = cv2.imread(img_path)
        #self.img = cv2.resize(self.img, (1920, 1080))
        self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def get_mask(self, img_file, img_file_annot):
        """
        Return the mask based on CIE-LAB color segmentation.
        The mask is used for filtering lines in detect()

        @type   file_annot: path
        @param  file_annot: path to annotated image file
        @rtype:   list
        @return:  the CIE-LAB mask as a list of length 2: (mean, std)
        """ 
        # Calculate a CIE-LAB color segmentation mask based on an annotated image (red pixel values)
        img_annot = cv2.imread(img_file_annot)
        img = cv2.imread(img_file)
        lower_limit = (0, 0, 245)
        upper_limit = (10, 10, 256)
        mask = cv2.inRange(img_annot,lower_limit, upper_limit)
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
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
            # Lines are in (x1, y1, x2, y2, width)
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
        #c = max(contours, key = lambda cnt: cv2.arcLength(cnt, closed=True))
        #contours_to_keep.append(c)
        areas = []
        # Use the contour with the largest bounding box in the image..
        # Should be the closest windmill
        for con in contours:
            #if cv2.contourArea(con) > 5000:
            #    contours_to_keep.append(con)
            x,y,w,h = cv2.boundingRect(con)
            areas.append(w*h)
        contours_to_keep.append(contours[np.argmax(areas)])
            
        # Fill mask with large contours
        cv2.drawContours(masked_image_zero, contours_to_keep, -1, 255, -1)
        # Use the new mask
        self.masked_image = masked_image_zero
        if self.save_result:
            cv2.imwrite(self.save_path[:-4] + '_contour_mask.jpg', self.masked_image)
        # Filter using the new mask
        self.filtered_lines = []
        for i in range(self.lines.shape[0]):
            pt1 = (int(self.lines[i, 0]), int(self.lines[i, 1]))
            pt2 = (int(self.lines[i, 2]), int(self.lines[i, 3]))
            width = self.lines[i, 4]
            if self.masked_image[pt1[1], pt1[0]] != 0 and self.masked_image[pt2[1], pt2[0]] != 0:
                cv2.line(self.img, pt1, pt2, (0,255,0), int(np.ceil(width / 2)))
                self.filtered_lines.append(self.lines[i])
        print("Number of unfiltered lines: {}".format(len(self.lines)))
        print("Number of filtered lines: {}".format(len(self.filtered_lines)))
        if self.save_result:
            cv2.imwrite(self.save_path[:-4] + '_filtered.jpg', self.img)
        self.estimate_best_fit()
        if self.save_result:
            cv2.imwrite(self.save_path[:-4] + '_final.jpg', self.img_final)

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
        best_score = np.inf
        for i in range(k):
            maybe_inliers = np.array(random.sample(self.filtered_lines, n))
            extreme_pts = self.find_extreme_points(self.masked_image)
            maybe_model_dict = self.get_turbine_model(maybe_inliers[:,:4], extreme_pts)
            # Empty dicts evaluates to False so True if not empty
            if maybe_model_dict:
                score = self.calculate_model_score(maybe_model_dict)
                if score < best_score:
                    best_score = score
                    best_fit = maybe_model_dict
        if self.save_result:
            cv2.imwrite(self.save_path[:-4] + '_intersections.jpg', self.img)
        print("Done!")
        print("Best score was: {}".format(best_score))
        if self.save_result:
            centroid = tuple(best_fit.get('centroid'))
            end_pts = best_fit.get('line_endpoints')
            cv2.circle(self.img_final, centroid, 8, (0,0,255), -1)
            for pt in end_pts:
                cv2.line(self.img_final, centroid, tuple(pt), (255,255,0), 2)

    def find_extreme_points(self, cnt):
        """
        Return the extreme points of the contour

        @type   cnt: contour/numpy array
        @param  cnt: Contour is a numpy array of values 0 or 255
        @rtype:   list of shape (4,2)
        @return:  the contour extreme points (left, right, top, bottom)
        """
        x, y, w, h = cv2.boundingRect(cnt)
        left = (x, np.argmax(cnt[:, x]))
        right = (x+w-1, np.argmax(cnt[:, x+w-1]))
        top = (np.argmax(self.masked_image[y, :]), y)
        bottom = (np.argmax(self.masked_image[y+h-1, :]), y+h-1)
        if self.save_result:
            cv2.circle(self.img, left, 8, (0, 50, 255), -1)
            cv2.circle(self.img, right, 8, (0, 255, 255), -1)
            cv2.circle(self.img, top, 8, (255, 50, 0), -1)
            cv2.circle(self.img, bottom, 8, (255, 255, 0), -1)
        return [left, right, top, bottom]

    def get_turbine_model(self, lines, extreme_pts):
        """
        Return the turbine model parameters given three lines and extreme points of contour.
        The turbine model parameters are: angle between lines, lengths,
        mean length, intersections, line_endpoints and centroid.
        The dict keys are: 'centroid', 'intersections', 'line_lengths',
        'mean_length', 'line_endpoints' and 'angles_between_lines'

        @type   lines: numpy array of shape (3,4)
        @param  lines: Each row is 4 numbers, x1, y1, x2, y2
        @type   extreme_pts: list of shape (4,2)
        @param  extreme_pts: the contour extreme points (left, right, top, bottom)
        @rtype:   dict
        @return:  dict containing the turbine model parameters
        """
        parameter_dict = {}
        lines_homogenous = []
        intersections = []
        for l in lines:
            # Transform each point (x,y) to homogenous coordinates (x,y,1)
            p1 = np.append(l[:2],1)
            p2 = np.append(l[2:],1)
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
        intersections = np.array([intersection[:2] / intersection[2] for intersection in intersections])

        # Calculate centroid of intersections (x,y) and round to int
        centroid = (np.sum(intersections, axis=0)/len(intersections)).astype(int)
        # Red dots on centroids if inside mask, else blue dots
        if centroid[0] >= 0 and centroid[1] >= 0 and centroid[1] < self.masked_image.shape[0] and centroid[0] < self.masked_image.shape[1]:
            if self.masked_image[centroid[1], centroid[0]] != 0:
                cv2.circle(self.img, tuple(centroid), 4, (0,0,255), thickness=-1)
                # Extend line from centroid to extreme left and right point
                #for i in range(len(lines_homogenous)):
                #    # Avoid dividing by 0
                #    if lines_homogenous[i,2] != 0:
                #        lines_homogenous[i,:] /= lines_homogenous[i,2]
                # Lines are a,b,c: ax+by+c = 0
                # y = -(a/b)x + c/b
                start_pt = tuple(centroid)
                end_pt_left = extreme_pts[0]
                end_pt_right = extreme_pts[1]
                #end_pt_bot = (1244,1563)
                end_pt_bot = (820, 530)
                # This is actually top, but keep name bot for convenience..
                #end_pt_bot = extreme_pts[2]
                parameter_dict["line_endpoints"] = [end_pt_left, end_pt_right, end_pt_bot]
                line_lengths = self.calculate_line_lengths(start_pt, [end_pt_left, end_pt_right, end_pt_bot])
                parameter_dict["line_lengths"] = line_lengths
                mean_length = np.sum(line_lengths)/len(line_lengths)
                parameter_dict["mean_length"] = mean_length
                angles_between_lines = self.calculate_angles_between_lines(start_pt, [end_pt_left, end_pt_right, end_pt_bot])
                parameter_dict["angles_between_lines"] = angles_between_lines
                parameter_dict["centroid"] = centroid
                parameter_dict["intersections"] = intersections
                return parameter_dict
            else:
                cv2.circle(self.img, tuple(centroid), 4, (255,0,0),thickness=-1)
        # Calculate length of each line,, mean length, angle between lines and we have what we need!
        return parameter_dict

    def dist_between_points(self, pt1, pt2):
        return sqrt((pt2[0]-pt1[0])*(pt2[0]-pt1[0]) + (pt2[1]-pt1[1])*(pt2[1]-pt1[1]))

    def angle_between_vectors(self, v1, v2):
        return np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))

    def calculate_line_lengths(self, centroid, points):
        """
        Return the line lengths given the centroid and end points.

        @type   centroid: tuple of length 2
        @param  centroid: Centroid is given in (x,y) coordinates
        @type   points: list of tuples
        @param  points: list of extreme points given as tuples in (x,y) coordinates
        @rtype:   numpy array of shape (,len(points))
        @return:  the length of each line from centroid to point in points
        """
        lengths = []
        for pt in points:
            lengths.append(self.dist_between_points(centroid, pt))
        return np.array(lengths)

    def calculate_angles_between_lines(self, centroid, points):
        """
        Return the angles between lines given the centroid and end points.

        @type   centroid: tuple of length 2
        @param  centroid: Centroid is given in (x,y) coordinates
        @type   points: list of tuples
        @param  points: list of extreme points given as tuples in (x,y) coordinates
        @rtype:   numpy array of shape (,len(points))
        @return:  the angle between each line pair (in degrees)
        """
        angles = []
        v1 = np.array((points[0][0] - centroid[0], points[0][1] - centroid[1]))
        v2 = np.array((points[1][0] - centroid[0], points[1][1] - centroid[1]))
        v3 = np.array((points[2][0] - centroid[0], points[2][1] - centroid[1]))
        angles.append(np.rad2deg(self.angle_between_vectors(v1, v2)))
        angles.append(np.rad2deg(self.angle_between_vectors(v1, v3)))
        angles.append(np.rad2deg(self.angle_between_vectors(v2, v3)))
        return np.array(angles)

    def calculate_model_score(self, model_dict):
        """
        Return the score based on the wind turbine model.
        The score is calculated from the model parameters, and the deviation
        of each intersection point from the centroid, the angle deviation from 120 deg.,
        and the deviation of each wing length from mean length.
        A small score is better.

        @type   model_dict: dict
        @param  model_dict: Dict containing the wind turbine model parameters
        @rtype:   float
        @return:  the score based on model parameters (lower is better)
        """
        intersections = model_dict.get('intersections')
        centroid = model_dict.get('centroid')
        angles_between_lines = model_dict.get('angles_between_lines')
        lengths = model_dict.get('line_lengths')
        mean_length = model_dict.get('mean_length')
        score = 0
        for i in range(3):
            int_i = intersections[i]
            a_i = angles_between_lines[i]
            length_i = lengths[i]
            score += np.dot((int_i - centroid),(int_i - centroid)) + (120 - a_i)*(120 - a_i) + (length_i - mean_length)*(length_i - mean_length)
        return score


def main():
    bd = BladeDetector(img_path='./scripts/image_data/gazebo.png', save_result=True)
    bd.get_mask('./scripts/image_data/offshore_wind_turbine.jpg', './scripts/image_data/annotated_wind_turbine.jpg')
    bd.detect()

if __name__ == "__main__":
    main()