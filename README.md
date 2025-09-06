🖼️ Instance Segmentation with Mask R-CNN (Python + OpenCV)

This project implements **Instance Segmentation** using the **Mask R-CNN deep learning model** with **Python** and **OpenCV**.  
It can detect and segment multiple objects in an image, highlight them with bounding boxes and colored masks, and display a detection summary.

🚀 Features
- Detects objects using **Mask R-CNN** trained on the **COCO dataset**.
- Generates **bounding boxes** with labels and confidence scores.
- Creates **segmentation masks** with random colors for each class.
- Displays total object counts and per-class counts.
- Provides two visualization windows:
  - Image with bounding boxes and labels.
  - Mask overlay visualization.


## 📂 Project Structure
```

├── module.py                 # Main script for instance segmentation
├── frozen\_inference\_graph.pb  # Pre-trained Mask R-CNN weights (download required)
├── mask\_rcnn\_inception\_v2\_coco\_2018\_01\_28.pbtxt # Model configuration
├── animals.jpg                # Sample test image

````

⚙️ Requirements
Make sure you have the following installed:

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy

Install dependencies with:
```bash
pip install opencv-python numpy
````

▶️ Usage
1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/Instance-Segmentation-MASK-R-CNN-with-Python-and-OpenCV.git
   cd Instance-Segmentation-MASK-R-CNN-with-Python-and-OpenCV
   ```

2. Download the pre-trained Mask R-CNN model files:
   * [`frozen_inference_graph_coco.pb`](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz)
   * `mask_rcnn_inception_v2_coco_2018_01_28.pbtxt`

3. Place the files in the project directory.

4. Run the script:

   ```bash
   python module.py
   ```
   
📊 Example Output
* Detected objects will be printed in the console with confidence scores.
* Two windows will open:
  * **Image with Boxes** → Original image with bounding boxes and labels.
  * **Mask Overlay** → Segmentation masks overlay.
    
📌 Notes
* The project uses **COCO dataset labels** (90 object categories).
* You can replace `animals.jpg` with any input image.
* Adjust the detection confidence threshold in `module.py` (default `0.3`) for stricter or more lenient detection.

📜 License

This project is licensed under the MIT License.

