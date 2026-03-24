import re
import time
import numpy as np
import cv2

class PlateRecognizer:
    def __init__(self):
        # Initialize EasyOCR reader (English)
        try:
            # Import lazily so app startup survives torch DLL/runtime issues.
            import easyocr
            # gpu=False or True depending on system availability. Using False for maximum compatibility.
            self.reader = easyocr.Reader(['en'], gpu=False)
        except Exception as e:
            print(f"Error loading EasyOCR: {e}")
            self.reader = None

    def execute_ocr(self, image_np):
        """
        Runs EasyOCR on a numpy array image and filters for potential license plates.
        Returns a list of extracted plate strings.
        """
        if self.reader is None:
            return ["OCR System Offline"]

        plates_found = []
        variants = self._build_ocr_variants(image_np)

        for variant in variants:
            results = self.reader.readtext(
                variant,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                decoder='greedy',
                paragraph=False,
                detail=1,
            )
            for (_, text, prob) in results:
                if prob < 0.22:
                    continue
                clean_text = self._normalize_plate_text(text)
                if self._looks_like_plate(clean_text):
                    plates_found.append(clean_text)
            if plates_found:
                break

        # Remove duplicates
        return list(set(plates_found))

    def detect_plate_regions(self, image_np):
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)
        grad_x = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1)
        grad_x = np.absolute(grad_x)
        min_val, max_val = grad_x.min(), grad_x.max()
        grad_x = 255 * ((grad_x - min_val) / (max(max_val - min_val, 1e-5)))
        grad_x = grad_x.astype("uint8")

        grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
        closed = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kernel)
        _, thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, square_kernel)
        thresh = cv2.erode(thresh, None, iterations=1)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:12]

        regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 40 or h < 14:
                continue
            ratio = w / float(h)
            area = w * h
            if 2.0 <= ratio <= 6.8 and area > 900:
                regions.append((x, y, w, h))

        return regions

    def run_plate_pipeline(self, image_np, candidate_boxes=None, max_seconds=1.25):
        start_time = time.perf_counter()
        regions = []
        extracted = []

        candidate_rois = self._candidate_plate_rois(image_np, candidate_boxes)
        if not candidate_rois:
            return {
                "plates": [],
                "regions": [],
            }

        for roi, offset_x, offset_y in candidate_rois[:4]:
            if (time.perf_counter() - start_time) > max_seconds:
                break

            local_regions = self.detect_plate_regions(roi)
            for (x, y, w, h) in local_regions[:3]:
                if (time.perf_counter() - start_time) > max_seconds:
                    break

                crop = roi[y:y+h, x:x+w]
                if crop.size == 0:
                    continue

                vals = self.execute_ocr(crop)
                for plate in vals:
                    if plate != "OCR System Offline":
                        extracted.append(plate)
                regions.append((x + offset_x, y + offset_y, w, h))

                if extracted:
                    break

            if extracted:
                break

        unique = []
        for value in extracted:
            normalized = re.sub(r'[^A-Z0-9]', '', value.upper())
            if 6 <= len(normalized) <= 12 and normalized not in unique:
                unique.append(normalized)

        return {
            "plates": unique,
            "regions": regions,
        }

    def _candidate_plate_rois(self, image_np, candidate_boxes):
        if not candidate_boxes:
            return []

        rois = []
        sorted_candidates = []
        for item in candidate_boxes:
            if isinstance(item, dict):
                x1, y1, x2, y2 = item.get("box", (0, 0, 0, 0))
            else:
                x1, y1, x2, y2 = item

            width = max(1, x2 - x1)
            height = max(1, y2 - y1)
            if width < 40 or height < 20:
                continue

            rx1 = max(0, x1 + int(width * 0.08))
            rx2 = min(image_np.shape[1], x2 - int(width * 0.08))
            ry1 = max(0, y1 + int(height * 0.42))
            ry2 = min(image_np.shape[0], y1 + int(height * 0.88))
            roi = image_np[ry1:ry2, rx1:rx2]
            if roi.size > 0:
                sorted_candidates.append((width * height, roi, rx1, ry1))
        sorted_candidates.sort(key=lambda item: item[0], reverse=True)
        for _, roi, rx1, ry1 in sorted_candidates[:4]:
            rois.append((roi, rx1, ry1))
        return rois

    def _build_ocr_variants(self, image_np):
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 45, 45)
        eq = cv2.equalizeHist(gray)
        thresh = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, 11)
        inv_thresh = cv2.bitwise_not(thresh)

        variants = []
        for item in [eq, thresh]:
            enlarged = cv2.resize(item, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
            variants.append(enlarged)
        return variants

    def _normalize_plate_text(self, text):
        clean = re.sub(r'[^A-Z0-9]', '', text.upper())
        if len(clean) < 6:
            return clean

        chars = list(clean)
        for idx in range(min(2, len(chars))):
            if chars[idx] == '0':
                chars[idx] = 'O'
            if chars[idx] == '1':
                chars[idx] = 'I'

        for idx in range(max(2, len(chars) - 4), len(chars)):
            if chars[idx] == 'O':
                chars[idx] = '0'
            if chars[idx] == 'I':
                chars[idx] = '1'
            if chars[idx] == 'Z':
                chars[idx] = '2'

        return ''.join(chars)

    def _looks_like_plate(self, text):
        if not text or len(text) < 6 or len(text) > 12:
            return False
        if text.isdigit() or text.isalpha():
            return False

        india_pattern = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{3,4}$')
        generic_pattern = re.compile(r'^[A-Z0-9]{6,12}$')
        return bool(india_pattern.match(text) or generic_pattern.match(text))
