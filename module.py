import cv2
import numpy as np
from collections import defaultdict

# Load Mask R-CNN
net = cv2.dnn.readNetFromTensorflow(
    "frozen_inference_graph_coco.pb",
    "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
)

# COCO class labels
class_names = [
    'person', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

colors = np.random.randint(0, 255, (len(class_names), 3))

img = cv2.imread("animals.jpg")
height, width, _ = img.shape

black_image = np.zeros((height, width, 3), dtype=np.uint8)
black_image[:] = (100, 100, 0)

blob = cv2.dnn.blobFromImage(img, swapRB=True)
net.setInput(blob)
boxes, masks = net.forward(["detection_out_final", "detection_masks"])
num_detections = boxes.shape[2]

object_counter = 0
object_names = defaultdict(int)

for i in range(num_detections):
    box = boxes[0, 0, i]
    score = box[2]
    if score < 0.3:
        continue

    class_id = int(box[1])
    if class_id >= len(class_names):
        continue

    x1 = max(0, int(box[3] * width))
    y1 = max(0, int(box[4] * height))
    x2 = min(width, int(box[5] * width))
    y2 = min(height, int(box[6] * height))

    roi_width = x2 - x1
    roi_height = y2 - y1
    if roi_width <= 0 or roi_height <= 0:
        continue

    mask = masks[i, class_id]
    mask = cv2.resize(mask, (roi_width, roi_height))
    mask = (mask > 0.3).astype(np.uint8)

    if cv2.countNonZero(mask) < 30:
        continue

    color = colors[class_id]
    roi = black_image[y1:y2, x1:x2]

    for c in range(3):
        roi[:, :, c] = np.where(mask == 1, color[c], roi[:, :, c])

    label = class_names[class_id]
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color.tolist(), 2)

    object_counter += 1
    object_names[label] += 1
    print(f"Detected: {label} ({score:.2f})")

# Summary on image
cv2.putText(img, f"Total Objects: {object_counter}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# Console summary
print("\n=== Detection Summary ===")
for name, count in object_names.items():
    print(f"{name}: {count}")
print(f"Total Objects: {object_counter}")

# Show images
cv2.imshow("Image with Boxes", img)
cv2.imshow("Mask Overlay", black_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
