import cv2
import numpy as np


class PreprocessingPipeline:
    @staticmethod
    def to_grayscale(image_rgb):
        if image_rgb.ndim == 2:
            return image_rgb
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def ensure_rgb(image):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image

    @staticmethod
    def apply_gaussian_blur(image, kernel_size=(5, 5)):
        return cv2.GaussianBlur(image, kernel_size, 0)

    @staticmethod
    def apply_median_filter(image, kernel_size=5):
        return cv2.medianBlur(image, kernel_size)

    @staticmethod
    def apply_bilateral_filter(image, diameter=9, sigma_color=75, sigma_space=75):
        return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)

    @staticmethod
    def apply_box_filter(image, kernel_size=5):
        return cv2.blur(image, (kernel_size, kernel_size))

    @staticmethod
    def histogram_equalization(image_gray):
        gray = PreprocessingPipeline.to_grayscale(image_gray)
        return cv2.equalizeHist(gray)

    @staticmethod
    def apply_clahe(image_gray, clip_limit=2.0, tile_grid_size=(8, 8)):
        gray = PreprocessingPipeline.to_grayscale(image_gray)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(gray)

    @staticmethod
    def gamma_correction(image_rgb, gamma=1.2):
        inv_gamma = 1.0 / max(gamma, 0.01)
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(image_rgb, table)

    @staticmethod
    def normalize_image(image_rgb):
        return cv2.normalize(image_rgb, None, 0, 255, cv2.NORM_MINMAX)

    @staticmethod
    def adaptive_thresholding(image_gray):
        gray = PreprocessingPipeline.to_grayscale(image_gray)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        )

    @staticmethod
    def threshold_variants(image_gray, threshold_value=127):
        gray = PreprocessingPipeline.to_grayscale(image_gray)
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        _, inverse = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
        _, trunc = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_TRUNC)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = PreprocessingPipeline.adaptive_thresholding(gray)
        return {
            "Binary Threshold": binary,
            "Adaptive Threshold": adaptive,
            "Otsu Threshold": otsu,
            "Inverse Threshold": inverse,
            "Trunc Threshold": trunc,
        }

    @staticmethod
    def morphological_operations(image_binary, operation="dilation", kernel_size=3):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if operation == "erosion":
            return cv2.erode(image_binary, kernel, iterations=1)
        if operation == "dilation":
            return cv2.dilate(image_binary, kernel, iterations=1)
        if operation == "opening":
            return cv2.morphologyEx(image_binary, cv2.MORPH_OPEN, kernel)
        if operation == "closing":
            return cv2.morphologyEx(image_binary, cv2.MORPH_CLOSE, kernel)
        if operation == "gradient":
            return cv2.morphologyEx(image_binary, cv2.MORPH_GRADIENT, kernel)
        if operation == "top_hat":
            return cv2.morphologyEx(image_binary, cv2.MORPH_TOPHAT, kernel)
        if operation == "black_hat":
            return cv2.morphologyEx(image_binary, cv2.MORPH_BLACKHAT, kernel)
        return image_binary

    @staticmethod
    def morphology_variants(image_gray, kernel_size=3):
        gray = PreprocessingPipeline.to_grayscale(image_gray)
        _, base = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return {
            "Input Binary": base,
            "Erosion": PreprocessingPipeline.morphological_operations(base, "erosion", kernel_size),
            "Dilation": PreprocessingPipeline.morphological_operations(base, "dilation", kernel_size),
            "Opening": PreprocessingPipeline.morphological_operations(base, "opening", kernel_size),
            "Closing": PreprocessingPipeline.morphological_operations(base, "closing", kernel_size),
            "Gradient": PreprocessingPipeline.morphological_operations(base, "gradient", kernel_size),
            "Top Hat": PreprocessingPipeline.morphological_operations(base, "top_hat", kernel_size),
            "Black Hat": PreprocessingPipeline.morphological_operations(base, "black_hat", kernel_size),
        }

    @staticmethod
    def _center_on_canvas(image_rgb, target_shape):
        target_h, target_w = target_shape[:2]
        src_h, src_w = image_rgb.shape[:2]
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        offset_x = max(0, (target_w - src_w) // 2)
        offset_y = max(0, (target_h - src_h) // 2)
        crop = image_rgb[: min(src_h, target_h), : min(src_w, target_w)]
        canvas[offset_y : offset_y + crop.shape[0], offset_x : offset_x + crop.shape[1]] = crop
        return canvas

    @staticmethod
    def transform_variants(image_rgb, scale=1.15, rotation=12, translate_x=25, translate_y=18):
        h, w = image_rgb.shape[:2]
        scaled = cv2.resize(image_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled = PreprocessingPipeline._center_on_canvas(scaled, image_rgb.shape)

        rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), rotation, 1.0)
        rotated = cv2.warpAffine(image_rgb, rotation_matrix, (w, h))

        translation_matrix = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
        translated = cv2.warpAffine(image_rgb, translation_matrix, (w, h))

        src_affine = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
        dst_affine = np.float32([[0.04 * w, 0.05 * h], [0.92 * w, 0.08 * h], [0.1 * w, 0.92 * h]])
        affine_matrix = cv2.getAffineTransform(src_affine, dst_affine)
        affine = cv2.warpAffine(image_rgb, affine_matrix, (w, h))

        src_perspective = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
        dst_perspective = np.float32([
            [0.08 * w, 0.04 * h],
            [0.92 * w, 0.12 * h],
            [0.03 * w, 0.94 * h],
            [0.97 * w, 0.88 * h],
        ])
        perspective_matrix = cv2.getPerspectiveTransform(src_perspective, dst_perspective)
        perspective = cv2.warpPerspective(image_rgb, perspective_matrix, (w, h))

        return {
            "Scaling": scaled,
            "Rotation": rotated,
            "Translation": translated,
            "Affine Transform": affine,
            "Perspective Transform": perspective,
        }

    @staticmethod
    def preprocessing_variants(image_rgb, kernel_size=5, gamma=1.2, clip_limit=2.0):
        gray = PreprocessingPipeline.to_grayscale(image_rgb)
        normalized = PreprocessingPipeline.normalize_image(image_rgb)
        return {
            "Original": image_rgb,
            "Grayscale": PreprocessingPipeline.ensure_rgb(gray),
            "Gaussian Blur": PreprocessingPipeline.ensure_rgb(PreprocessingPipeline.apply_gaussian_blur(gray, (kernel_size, kernel_size))),
            "Median Blur": PreprocessingPipeline.ensure_rgb(PreprocessingPipeline.apply_median_filter(gray, kernel_size)),
            "Bilateral Filtering": PreprocessingPipeline.ensure_rgb(PreprocessingPipeline.apply_bilateral_filter(gray)),
            "Box Filter": PreprocessingPipeline.ensure_rgb(PreprocessingPipeline.apply_box_filter(gray, kernel_size)),
            "Histogram Equalization": PreprocessingPipeline.ensure_rgb(PreprocessingPipeline.histogram_equalization(gray)),
            "CLAHE": PreprocessingPipeline.ensure_rgb(PreprocessingPipeline.apply_clahe(gray, clip_limit=clip_limit)),
            "Gamma Correction": PreprocessingPipeline.gamma_correction(image_rgb, gamma=gamma),
            "Normalization": normalized,
        }

    @staticmethod
    def get_all_stages(image_rgb):
        stages = {}
        stages.update(PreprocessingPipeline.preprocessing_variants(image_rgb))
        for name, image in PreprocessingPipeline.threshold_variants(image_rgb).items():
            stages[name] = PreprocessingPipeline.ensure_rgb(image)
        for name, image in PreprocessingPipeline.morphology_variants(image_rgb).items():
            stages[f"Morph {name}"] = PreprocessingPipeline.ensure_rgb(image)
        return stages
