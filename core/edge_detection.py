import cv2
import numpy as np


class EdgeDetector:
    @staticmethod
    def _normalize_response(image):
        normalized = cv2.normalize(np.abs(image), None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)

    @staticmethod
    def apply_sobel(image_gray, kernel_size=3):
        sobelx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobely = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
        return cv2.convertScaleAbs(cv2.magnitude(sobelx, sobely))

    @staticmethod
    def sobel_components(image_gray, kernel_size=3):
        gx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        gy = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
        magnitude = cv2.magnitude(gx, gy)
        direction = cv2.phase(gx, gy, angleInDegrees=True)
        return (
            EdgeDetector._normalize_response(gx),
            EdgeDetector._normalize_response(gy),
            EdgeDetector._normalize_response(magnitude),
            EdgeDetector._normalize_response(direction),
        )

    @staticmethod
    def apply_prewitt(image_gray):
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        px = cv2.filter2D(image_gray, cv2.CV_32F, kernelx)
        py = cv2.filter2D(image_gray, cv2.CV_32F, kernely)
        return EdgeDetector._normalize_response(cv2.magnitude(px, py))

    @staticmethod
    def apply_roberts(image_gray):
        kernelx = np.array([[1, 0], [0, -1]], dtype=np.float32)
        kernely = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        rx = cv2.filter2D(image_gray, cv2.CV_32F, kernelx)
        ry = cv2.filter2D(image_gray, cv2.CV_32F, kernely)
        return EdgeDetector._normalize_response(cv2.magnitude(rx, ry))

    @staticmethod
    def apply_laplacian(image_gray, kernel_size=3):
        laplacian = cv2.Laplacian(image_gray, cv2.CV_64F, ksize=kernel_size)
        return cv2.convertScaleAbs(laplacian)

    @staticmethod
    def apply_canny(image_gray, lower=50, upper=150, sigma=1.0):
        blur_size = 5 if sigma <= 1.2 else 7
        blurred = cv2.GaussianBlur(image_gray, (blur_size, blur_size), sigma)
        return cv2.Canny(blurred, lower, upper)

    @staticmethod
    def apply_scharr(image_gray):
        scharrx = cv2.Scharr(image_gray, cv2.CV_64F, 1, 0)
        scharry = cv2.Scharr(image_gray, cv2.CV_64F, 0, 1)
        return cv2.convertScaleAbs(cv2.magnitude(scharrx, scharry))

    @staticmethod
    def apply_kirsch(image_gray):
        kernels = [
            np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], dtype=np.float32),
            np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]], dtype=np.float32),
            np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]], dtype=np.float32),
            np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]], dtype=np.float32),
            np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], dtype=np.float32),
            np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]], dtype=np.float32),
            np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], dtype=np.float32),
            np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]], dtype=np.float32),
        ]
        responses = [cv2.filter2D(image_gray, cv2.CV_32F, kernel) for kernel in kernels]
        return EdgeDetector._normalize_response(np.max(np.stack(responses, axis=0), axis=0))

    @staticmethod
    def apply_robinson_compass(image_gray):
        base_kernels = [
            np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32),
            np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype=np.float32),
            np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
            np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=np.float32),
        ]
        responses = [cv2.filter2D(image_gray, cv2.CV_32F, kernel) for kernel in base_kernels]
        responses.extend([-response for response in responses])
        return EdgeDetector._normalize_response(np.max(np.stack(responses, axis=0), axis=0))

    @staticmethod
    def apply_frei_chen(image_gray):
        root2 = np.sqrt(2)
        kernelx = np.array([[1, root2, 1], [0, 0, 0], [-1, -root2, -1]], dtype=np.float32)
        kernely = np.array([[-1, 0, 1], [-root2, 0, root2], [-1, 0, 1]], dtype=np.float32)
        fx = cv2.filter2D(image_gray, cv2.CV_32F, kernelx)
        fy = cv2.filter2D(image_gray, cv2.CV_32F, kernely)
        return EdgeDetector._normalize_response(cv2.magnitude(fx, fy))

    @staticmethod
    def apply_dog(image_gray, sigma=1.2):
        sigma_small = max(0.6, sigma)
        sigma_large = sigma_small * 1.6
        blur_small = cv2.GaussianBlur(image_gray, (0, 0), sigma_small)
        blur_large = cv2.GaussianBlur(image_gray, (0, 0), sigma_large)
        dog = cv2.subtract(blur_small, blur_large)
        return EdgeDetector._normalize_response(dog)

    @staticmethod
    def gradient_magnitude(image_gray, kernel_size=3):
        gx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        gy = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
        mag = cv2.magnitude(gx, gy)
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return mag_norm.astype(np.uint8)

    @staticmethod
    def edge_overlay(image_rgb, edge_gray):
        overlay = image_rgb.copy()
        overlay[edge_gray > 0] = [0, 247, 255]
        return overlay

    @staticmethod
    def all_algorithms(image_rgb, lower=50, upper=150, kernel_size=3, sigma=1.0):
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        gx, gy, magnitude, direction = EdgeDetector.sobel_components(gray, kernel_size=kernel_size)
        canny = EdgeDetector.apply_canny(gray, lower=lower, upper=upper, sigma=sigma)
        return {
            "Original": image_rgb,
            "Sobel": cv2.cvtColor(EdgeDetector.apply_sobel(gray, kernel_size), cv2.COLOR_GRAY2RGB),
            "Prewitt": cv2.cvtColor(EdgeDetector.apply_prewitt(gray), cv2.COLOR_GRAY2RGB),
            "Roberts": cv2.cvtColor(EdgeDetector.apply_roberts(gray), cv2.COLOR_GRAY2RGB),
            "Laplacian": cv2.cvtColor(EdgeDetector.apply_laplacian(gray, kernel_size), cv2.COLOR_GRAY2RGB),
            "Canny": cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB),
            "Scharr": cv2.cvtColor(EdgeDetector.apply_scharr(gray), cv2.COLOR_GRAY2RGB),
            "Kirsch": cv2.cvtColor(EdgeDetector.apply_kirsch(gray), cv2.COLOR_GRAY2RGB),
            "Robinson Compass": cv2.cvtColor(EdgeDetector.apply_robinson_compass(gray), cv2.COLOR_GRAY2RGB),
            "Frei-Chen": cv2.cvtColor(EdgeDetector.apply_frei_chen(gray), cv2.COLOR_GRAY2RGB),
            "Difference of Gaussian": cv2.cvtColor(EdgeDetector.apply_dog(gray, sigma=sigma), cv2.COLOR_GRAY2RGB),
            "Gradient X": cv2.cvtColor(gx, cv2.COLOR_GRAY2RGB),
            "Gradient Y": cv2.cvtColor(gy, cv2.COLOR_GRAY2RGB),
            "Magnitude": cv2.cvtColor(magnitude, cv2.COLOR_GRAY2RGB),
            "Direction": cv2.cvtColor(direction, cv2.COLOR_GRAY2RGB),
            "Edge Overlay": EdgeDetector.edge_overlay(image_rgb, canny),
        }

    @staticmethod
    def detect_edges(image_rgb, method="Canny", lower=50, upper=150, kernel_size=3, sigma=1.0):
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        method_name = method.lower()
        if method_name == "sobel":
            edges = EdgeDetector.apply_sobel(gray, kernel_size=kernel_size)
        elif method_name == "prewitt":
            edges = EdgeDetector.apply_prewitt(gray)
        elif method_name == "roberts":
            edges = EdgeDetector.apply_roberts(gray)
        elif method_name == "laplacian":
            edges = EdgeDetector.apply_laplacian(gray, kernel_size=kernel_size)
        elif method_name == "scharr":
            edges = EdgeDetector.apply_scharr(gray)
        elif method_name == "kirsch":
            edges = EdgeDetector.apply_kirsch(gray)
        elif method_name == "robinson":
            edges = EdgeDetector.apply_robinson_compass(gray)
        elif method_name == "frei-chen":
            edges = EdgeDetector.apply_frei_chen(gray)
        elif method_name == "dog":
            edges = EdgeDetector.apply_dog(gray, sigma=sigma)
        else:
            edges = EdgeDetector.apply_canny(gray, lower=lower, upper=upper, sigma=sigma)

        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
