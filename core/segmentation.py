import cv2
import numpy as np


class SegmentationEngine:
    @staticmethod
    def threshold_segmentation(image_rgb):
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        # Apply Otsu's thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        occupancy = float(np.mean(thresh > 0))
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB), occupancy

    @staticmethod
    def watershed_segmentation(image_rgb):
        # Convert to grayscale
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal using morphological opening
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
        
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        
        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0
        
        # Apply Watershed
        img_copy = image_rgb.copy()
        cv2.watershed(img_copy, markers)
        
        # Mark boundaries in red
        img_copy[markers == -1] = [255, 0, 0]
        
        return img_copy, markers

    @staticmethod
    def region_based_segmentation(image_rgb):
        filtered = cv2.pyrMeanShiftFiltering(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), 21, 51)
        return cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)

    @staticmethod
    def kmeans_segmentation(image_rgb, clusters=4):
        pixel_vals = image_rgb.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
        _, labels, centers = cv2.kmeans(pixel_vals, clusters, None, criteria, 8, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented = centers[labels.flatten()].reshape(image_rgb.shape)
        return segmented, centers

    @staticmethod
    def color_based_segmentation(image_rgb):
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        sat_mask = cv2.inRange(hsv, np.array([0, 40, 30]), np.array([180, 255, 255]))
        result = np.zeros_like(image_rgb)
        result[sat_mask > 0] = image_rgb[sat_mask > 0]
        return result, sat_mask

    @staticmethod
    def full_segmentation(image_rgb):
        threshold_img, occupancy = SegmentationEngine.threshold_segmentation(image_rgb)
        region_img = SegmentationEngine.region_based_segmentation(image_rgb)
        kmeans_img, centers = SegmentationEngine.kmeans_segmentation(image_rgb)
        color_img, color_mask = SegmentationEngine.color_based_segmentation(image_rgb)
        watershed_img, markers = SegmentationEngine.watershed_segmentation(image_rgb)
        return {
            "Threshold Segmentation": threshold_img,
            "Region Segmentation": region_img,
            "Watershed Segmentation": watershed_img,
            "K-Means Segmentation": kmeans_img,
            "Color-Based Segmentation": color_img,
            "lane_occupancy": round(occupancy, 3),
            "watershed_regions": int(np.max(markers)) if markers is not None else 0,
            "kmeans_clusters": int(len(centers)) if centers is not None else 0,
            "color_mask_coverage": round(float(np.mean(color_mask > 0)), 3),
        }
