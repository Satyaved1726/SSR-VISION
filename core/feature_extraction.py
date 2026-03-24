import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure


class FeatureExtractor:
    @staticmethod
    def extract_hog(image_rgb):
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Extract HOG features and HOG image for visualization
        fd, hog_image = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualize=True, channel_axis=None)
        
        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        
        # Convert to 8-bit for UI
        hog_image_8bit = (hog_image_rescaled * 255).astype(np.uint8)
        return cv2.cvtColor(hog_image_8bit, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def extract_orb_keypoints(image_rgb):
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create(nfeatures=600)
        keypoints = orb.detect(gray, None)
        keypoints, _ = orb.compute(gray, keypoints)
        kp_count = len(keypoints) if keypoints is not None else 0
        img2 = cv2.drawKeypoints(image_rgb, keypoints, None, color=(0, 243, 255), flags=0)
        return img2, kp_count

    @staticmethod
    def extract_sift_keypoints(image_rgb):
        if not hasattr(cv2, "SIFT_create"):
            return image_rgb.copy(), 0, False
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create()
        keypoints, _ = sift.detectAndCompute(gray, None)
        count = len(keypoints) if keypoints is not None else 0
        output = cv2.drawKeypoints(image_rgb, keypoints, None, color=(255, 183, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return output, count, True

    @staticmethod
    def extract_surf_keypoints(image_rgb):
        if not hasattr(cv2, "xfeatures2d") or not hasattr(cv2.xfeatures2d, "SURF_create"):
            return image_rgb.copy(), 0, False
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        surf = cv2.xfeatures2d.SURF_create(400)
        keypoints, _ = surf.detectAndCompute(gray, None)
        count = len(keypoints) if keypoints is not None else 0
        output = cv2.drawKeypoints(image_rgb, keypoints, None, color=(255, 64, 64), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return output, count, True

    @staticmethod
    def detect_contours(image_rgb):
        img_copy = image_rgb.copy()
        gray = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_copy, contours, -1, (0, 243, 255), 2)
        return img_copy, len(contours)

    @staticmethod
    def detect_blobs(image_rgb):
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 40
        params.maxArea = 20000
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)
        output = cv2.drawKeypoints(image_rgb, keypoints, None, (0, 255, 154), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return output, len(keypoints)

    @staticmethod
    def shape_recognition(image_rgb):
        output = image_rgb.copy()
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        stats = {"Triangle": 0, "Quadrilateral": 0, "Circle": 0, "Irregular": 0}
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 120:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)

            if len(approx) == 3:
                label = "Triangle"
            elif len(approx) == 4:
                label = "Quadrilateral"
            elif len(approx) > 7:
                label = "Circle"
            else:
                label = "Irregular"

            stats[label] += 1
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 183, 0), 2)
            cv2.putText(output, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 183, 0), 2)

        return output, stats

    @staticmethod
    def classify_color(rgb_triplet):
        red, green, blue = [int(v) for v in rgb_triplet]
        if max(red, green, blue) < 45:
            return "Black"
        if min(red, green, blue) > 220:
            return "White"
        if abs(red - green) < 18 and abs(green - blue) < 18:
            return "Gray"
        if red >= green and red >= blue:
            return "Red"
        if green >= red and green >= blue:
            return "Green"
        return "Blue"

    @staticmethod
    def analyze_color_profile(image_rgb):
        sample = image_rgb.reshape(-1, 3)
        if sample.shape[0] > 12000:
            indices = np.linspace(0, sample.shape[0] - 1, 12000, dtype=int)
            sample = sample[indices]
        sample = sample.astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(sample, 3, None, criteria, 6, cv2.KMEANS_PP_CENTERS)
        centers = np.uint8(centers)
        counts = np.bincount(labels.flatten(), minlength=len(centers))
        dominant_center = centers[int(np.argmax(counts))]
        histograms = {
            "Red": cv2.calcHist([image_rgb], [0], None, [32], [0, 256]).flatten(),
            "Green": cv2.calcHist([image_rgb], [1], None, [32], [0, 256]).flatten(),
            "Blue": cv2.calcHist([image_rgb], [2], None, [32], [0, 256]).flatten(),
        }
        return {
            "dominant_rgb": dominant_center.tolist(),
            "dominant_color": FeatureExtractor.classify_color(dominant_center),
            "histograms": histograms,
        }

    @staticmethod
    def full_feature_pack(image_rgb):
        hog_img = FeatureExtractor.extract_hog(image_rgb)
        orb_img, orb_count = FeatureExtractor.extract_orb_keypoints(image_rgb)
        sift_img, sift_count, sift_available = FeatureExtractor.extract_sift_keypoints(image_rgb)
        surf_img, surf_count, surf_available = FeatureExtractor.extract_surf_keypoints(image_rgb)
        contour_img, contour_count = FeatureExtractor.detect_contours(image_rgb)
        blob_img, blob_count = FeatureExtractor.detect_blobs(image_rgb)
        shape_img, shape_stats = FeatureExtractor.shape_recognition(image_rgb)
        color_profile = FeatureExtractor.analyze_color_profile(image_rgb)
        return {
            "HOG Descriptors": hog_img,
            "ORB Keypoints": orb_img,
            "SIFT Keypoints": sift_img,
            "SURF Keypoints": surf_img,
            "Contours": contour_img,
            "Blob Detection": blob_img,
            "Shape Recognition": shape_img,
            "orb_count": orb_count,
            "sift_count": sift_count,
            "sift_available": sift_available,
            "surf_count": surf_count,
            "surf_available": surf_available,
            "blob_count": blob_count,
            "contour_count": contour_count,
            "shape_stats": shape_stats,
            "color_profile": color_profile,
        }
