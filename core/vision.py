import cv2
import numpy as np
from collections import Counter

# Load YOLOv8 model (nano for speed, will download automatically if missing)
# Filter for key traffic classes in COCO: 0 (person), 1 (bicycle), 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
TRAFFIC_CLASSES = [0, 1, 2, 3, 5, 7]
CLASS_NAMES = {0: 'Pedestrian', 1: 'Bicycle', 2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}
VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}

class VisionAnalyzer:
    def __init__(self):
        try:
            # Import lazily so app startup survives torch DLL/runtime issues.
            from ultralytics import YOLO
            self.model = YOLO('yolov8n.pt')
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None

    def analyze_image(self, image_pil):
        """
        Main analysis pipeline for a traffic image.
        Returns processed image (RGB numpy) and a dict of results.
        """
        # Convert PIL to CV2 format (BGR)
        img_np = np.array(image_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        results = {
            "vehicle_count": 0,
            "pedestrian_count": 0,
            "density_level": "LOW",
            "violations": [],
            "road_condition": "NORMAL",
            "bboxes": [],
            "vehicles": [],
            "type_distribution": {},
            "lane_occupancy": 0.0,
            "vehicle_spacing": 0.0,
            "detected_plates": [],
            "helmet_checks": [],
        }

        if self.model is None:
            return img_np, results

        # Run YOLO inference
        detections = self.model(img_bgr, classes=TRAFFIC_CLASSES, conf=0.25)[0]
        
        vehicles = []
        pedestrians = []
        centers = []
        for box in detections.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            label = CLASS_NAMES.get(cls_id, "Vehicle")
            area = max(1, (x2 - x1) * (y2 - y1))
            if cls_id in VEHICLE_CLASS_IDS:
                crop = img_bgr[y1:y2, x1:x2]
                vehicle_color = self._estimate_vehicle_color(crop)
                vehicle_type = self._normalize_vehicle_type(label)
                vehicle_model = self._estimate_vehicle_model(vehicle_type, vehicle_color, x2 - x1, y2 - y1)

                vehicles.append({
                    "label": label,
                    "vehicle_type": vehicle_type,
                    "vehicle_color": vehicle_color,
                    "vehicle_model": vehicle_model,
                    "box": (x1, y1, x2, y2),
                    "conf": conf,
                    "cls": cls_id,
                    "area": area,
                })
                centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
            elif cls_id == 0:
                pedestrians.append({
                    "label": "Pedestrian",
                    "box": (x1, y1, x2, y2),
                    "conf": conf,
                    "cls": cls_id,
                    "area": area,
                })
            
            # Draw bounding box (Cyan/Neon Blue style)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 243, 0), 2)
            cv2.putText(img_bgr, f"{label} {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 243, 0), 2)

        results["vehicle_count"] = len(vehicles)
        results["pedestrian_count"] = len(pedestrians)
        results["bboxes"] = vehicles
        results["vehicles"] = vehicles
        results["type_distribution"] = dict(Counter(v["vehicle_type"] for v in vehicles))

        # 2. Traffic Density Estimation
        lane_occupancy = 0.0
        if img_bgr.shape[0] > 0 and img_bgr.shape[1] > 0:
            total_area = img_bgr.shape[0] * img_bgr.shape[1]
            occupied_area = sum(v["area"] for v in vehicles)
            lane_occupancy = occupied_area / total_area
        results["lane_occupancy"] = round(lane_occupancy, 3)

        spacing_metric = self._average_spacing(centers)
        results["vehicle_spacing"] = round(spacing_metric, 2)

        if results["vehicle_count"] > 22 or lane_occupancy > 0.35 or results["pedestrian_count"] > 12:
            results["density_level"] = "CRITICAL"
        elif results["vehicle_count"] > 14 or lane_occupancy > 0.25 or results["pedestrian_count"] > 8:
            results["density_level"] = "HIGH"
        elif results["vehicle_count"] > 6 or lane_occupancy > 0.12 or results["pedestrian_count"] > 3:
            results["density_level"] = "MEDIUM"
        else:
            results["density_level"] = "LOW"

        # 3. Violation Detection
        # Only emit violations that we can evaluate conservatively to avoid false positives.
        rider_pairs = self._pair_motorcycles_and_riders(vehicles, pedestrians)
        for motorcycle, rider in rider_pairs:
            helmet_absent, confidence = self._detect_helmet_absence(img_bgr, rider["box"])
            results["helmet_checks"].append({
                "motorcycle_box": motorcycle["box"],
                "rider_box": rider["box"],
                "helmet_absent": helmet_absent,
                "confidence": confidence,
            })
            if helmet_absent:
                results["violations"].append("HELMET ABSENCE DETECTED")
                x1, y1, x2, y2 = motorcycle["box"]
                rx1, ry1, rx2, ry2 = rider["box"]
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.rectangle(img_bgr, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
                cv2.putText(img_bgr, f"NO HELMET {confidence:.2f}", (x1, max(25, y1 - 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 4. Road Condition Detection (OpenCV edges variance on bottom half)
        from core.edge_detection import EdgeDetector
        h, w = img_bgr.shape[:2]
        road_roi = img_bgr[int(h*0.6):h, :] # bottom 40%
        if road_roi.size > 0:
            gray = cv2.cvtColor(road_roi, cv2.COLOR_BGR2GRAY)
            edges = EdgeDetector.apply_canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            if edge_density > 0.04:
                results["road_condition"] = "DAMAGED/OBSTRUCTED"
            elif edge_density < 0.01:
                 results["road_condition"] = "FADED MARKINGS"
            else:
                 results["road_condition"] = "GOOD"

        # Convert back to RGB for Streamlit display
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb, results

    def _estimate_vehicle_color(self, crop_bgr):
        if crop_bgr is None or crop_bgr.size == 0:
            return "Unknown"
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        mean_h, mean_s, mean_v = np.mean(hsv.reshape(-1, 3), axis=0)
        if mean_v < 45:
            return "Black"
        if mean_v > 200 and mean_s < 40:
            return "White"
        if mean_s < 50:
            return "Gray"
        if mean_h < 12 or mean_h > 165:
            return "Red"
        if mean_h < 28:
            return "Yellow"
        if mean_h < 85:
            return "Green"
        if mean_h < 130:
            return "Blue"
        return "Other"

    def _normalize_vehicle_type(self, label):
        mapped = {
            "Car": "Car",
            "Truck": "Truck",
            "Bus": "Bus",
            "Bicycle": "Bicycle",
            "Motorcycle": "Bike",
        }
        return mapped.get(label, "Vehicle")

    def _estimate_vehicle_model(self, vehicle_type, color, width, height):
        # Lightweight demo-only model estimate, useful for intelligence panel previews.
        area = width * height
        if vehicle_type == "Bus":
            return "City Transit Class"
        if vehicle_type == "Truck":
            return "Commercial Hauler"
        if vehicle_type == "Bike":
            return "Commuter Bike"
        if vehicle_type == "Bicycle":
            return "Urban Bicycle"
        if vehicle_type == "Car":
            if area > 45000:
                return "SUV Class"
            if color in ["White", "Gray"]:
                return "Sedan Class"
            return "Hatchback Class"
        return "Unknown Model"

    def _average_spacing(self, centers):
        if len(centers) < 2:
            return 0.0
        distances = []
        for i in range(1, len(centers)):
            x1, y1 = centers[i - 1]
            x2, y2 = centers[i]
            distances.append(float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)))
        return float(np.mean(distances)) if distances else 0.0

    def _pair_motorcycles_and_riders(self, vehicles, pedestrians):
        pairs = []
        motorcycles = [item for item in vehicles if item.get("cls") == 3]
        for motorcycle in motorcycles:
            mx1, my1, mx2, my2 = motorcycle["box"]
            best_match = None
            best_score = 0.0
            for person in pedestrians:
                px1, py1, px2, py2 = person["box"]
                horizontal_overlap = max(0, min(mx2, px2) - max(mx1, px1))
                vertical_overlap = max(0, min(my2, py2) - max(my1, py1))
                person_width = max(1, px2 - px1)
                person_height = max(1, py2 - py1)
                overlap_ratio = horizontal_overlap / float(person_width)
                vertical_alignment = vertical_overlap / float(person_height)
                rider_above_bike = py1 < my2 and py2 > my1
                score = overlap_ratio + (0.35 if rider_above_bike else 0.0) + (0.2 if vertical_alignment > 0.2 else 0.0)
                if score > best_score and overlap_ratio > 0.25:
                    best_score = score
                    best_match = person
            if best_match is not None:
                pairs.append((motorcycle, best_match))
        return pairs

    def _detect_helmet_absence(self, image_bgr, person_box):
        x1, y1, x2, y2 = person_box
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        if width < 18 or height < 40:
            return False, 0.0

        head_y2 = y1 + max(12, int(height * 0.32))
        head_x1 = x1 + int(width * 0.12)
        head_x2 = x2 - int(width * 0.12)
        head_roi = image_bgr[y1:head_y2, head_x1:head_x2]
        if head_roi.size == 0:
            return False, 0.0

        hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)

        lower_skin1 = np.array([0, 25, 60], dtype=np.uint8)
        upper_skin1 = np.array([20, 180, 255], dtype=np.uint8)
        lower_skin2 = np.array([160, 25, 60], dtype=np.uint8)
        upper_skin2 = np.array([179, 180, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin1, upper_skin1) | cv2.inRange(hsv, lower_skin2, upper_skin2)
        skin_ratio = float(np.count_nonzero(skin_mask)) / float(skin_mask.size)

        dark_ratio = float(np.count_nonzero(gray < 70)) / float(gray.size)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = float(np.count_nonzero(edges)) / float(edges.size)

        confidence = (skin_ratio * 1.4) + (max(0.0, 0.45 - dark_ratio) * 0.6) + (edge_ratio * 0.25)
        helmet_absent = skin_ratio > 0.18 and dark_ratio < 0.35 and confidence > 0.32
        return helmet_absent, round(float(min(confidence, 0.99)), 2)
