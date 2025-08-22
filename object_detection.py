import numpy as np
import csv
import datetime
import cv2
from ultralytics import YOLO
import easyocr

# Load the YOLOv8n model with tracking
model = YOLO('yolov8n.pt')
reader = easyocr.Reader(['en'])

def detect_traffic_light(image, light_roi):
    """
    Detect traffic light state (red or green) in the specified ROI.
    light_roi: Tuple of ((x1, y1), (x2, y2)) defining the traffic light ROI.
    Returns: 'red', 'green', or 'unknown'.
    """
    if light_roi is None:
        return 'unknown'

    (x1, y1), (x2, y2) = light_roi
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w-1))
    x2 = max(0, min(x2, w-1))
    y1 = max(0, min(y1, h-1))
    y2 = max(0, min(y2, h-1))
    
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return 'unknown'
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red_lower1 = np.array([0, 70, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 70, 50])
    red_upper2 = np.array([180, 255, 255])
    green_lower = np.array([35, 50, 50])
    green_upper = np.array([85, 255, 255])
    
    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_green = cv2.inRange(hsv, green_lower, green_upper)
    
    red_pixels = cv2.countNonZero(mask_red)
    green_pixels = cv2.countNonZero(mask_green)
    total_pixels = roi.shape[0] * roi.shape[1]
    
    if red_pixels > 0.1 * total_pixels:
        return 'red'
    elif green_pixels > 0.1 * total_pixels:
        return 'green'
    return 'unknown'

def intersection(p, q, r, t):
    (x1, y1) = p
    (x2, y2) = q
    (x3, y3) = r
    (x4, y4) = t

    a1 = y1-y2
    b1 = x2-x1
    c1 = x1*y2-x2*y1

    a2 = y3-y4
    b2 = x4-x3
    c2 = x3*y4-x4*y3

    if(a1*b2-a2*b1 == 0):
        return False
    x = (b1*c2 - b2*c1) / (a1*b2 - a2*b1)
    y = (a2*c1 - a1*c2) / (a1*b2 - a2*b1)

    if x1 > x2:
        tmp = x1
        x1 = x2
        x2 = tmp
    if y1 > y2:
        tmp = y1
        y1 = y2
        y2 = tmp
    if x3 > x4:
        tmp = x3
        x3 = x4
        x4 = tmp
    if y3 > y4:
        tmp = y3
        y3 = y4
        y4 = tmp

    if x >= x1 and x <= x2 and y >= y1 and y <= y2 and x >= x3 and x <= x4 and y >= y3 and y <= y4:
        return True
    return False

def draw_boxes(image, boxes, line, labels, obj_thresh, dcnt, light_roi=None):
    with open(r'C:\Users\n2007\Traffic-Signal-Violation-Detection-System\Resources\violations.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Vehicle_ID', 'Track_ID', 'License_Plate', 'Timestamp'])

    def log_violation(vehicle_id, track_id, license_plate):
        with open(r'C:\Users\n2007\Traffic-Signal-Violation-Detection-System\Resources\violations.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([vehicle_id, track_id, license_plate, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')])

    light_state = detect_traffic_light(image, light_roi)
    print(f"Traffic light state: {light_state}")

    for box in boxes:
        label_str = labels[int(box.cls.cpu().numpy())]
        score = box.conf.cpu().numpy().item()
        track_id = int(box.id.cpu().numpy().item()) if box.id is not None else dcnt
        
        # Print and display label and confidence for all detected vehicles
        print(f"Detected: {label_str}, Confidence: {score*100:.2f}%, Track ID: {track_id}")
        cv2.putText(image, 
                    f"{label_str} {score:.2f}", 
                    (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    2)

        if score > obj_thresh:
            rxmin, rymin, rxmax, rymax = box.xyxy[0].cpu().numpy().astype(int)
            tf = False

            tf |= intersection(line[0], line[1], (rxmin, rymin), (rxmin, rymax))
            tf |= intersection(line[0], line[1], (rxmax, rymin), (rxmax, rymax))
            tf |= intersection(line[0], line[1], (rxmin, rymin), (rxmax, rymin))
            tf |= intersection(line[0], line[1], (rxmin, rymax), (rxmax, rymax))

            cv2.line(image, line[0], line[1], (255, 0, 0), 3)

            if tf and light_state == 'red':
                cv2.rectangle(image, (rxmin, rymin), (rxmax, rymax), (255, 0, 0), 3)
                cimg = image[rymin:rymax, rxmin:rxmax]
                violation_path = f"C:/Users/n2007/Traffic-Signal-Violation-Detection-System/Detected Images/violation_{track_id}.jpg"
                cv2.imwrite(violation_path, cimg)
                
                ocr_result = reader.readtext(cimg, detail=0)
                license_plate = ' '.join(ocr_result) if ocr_result else 'Unknown'
                print(f"Violation - License Plate: {license_plate}")
                
                log_violation(f"Vehicle_{track_id}", track_id, license_plate)
                dcnt += 1
            else:
                cv2.rectangle(image, (rxmin, rymin), (rxmax, rymax), (0, 255, 0), 3)

            cv2.putText(image, 
                        f"ID:{track_id}", 
                        (rxmin, rymin - 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 255, 0), 
                        2)
        
    return image, dcnt

def process_image(image_path, line, labels, obj_thresh=0.5, nms_thresh=0.45, light_roi=None):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")

    results = model(img, conf=obj_thresh, iou=nms_thresh)
    detections = results[0].boxes
    output_img = img.copy()

    output_img, dcnt = draw_boxes(output_img, detections, line, labels, obj_thresh, dcnt=0, light_roi=light_roi)
    return output_img

labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
          "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
          "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
          "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]