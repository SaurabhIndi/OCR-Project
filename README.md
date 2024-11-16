# Hybrid OCR System Documentation

## Overview
This project implements a **Hybrid OCR System** that combines **template matching** and **CNN-based OCR** (using PaddleOCR) for robust text recognition. The system leverages template matching for predictable fonts (e.g., digital displays) and falls back to CNN-based OCR for more complex cases (e.g., CAPTCHAs). 

---

## Features
1. **Template Matching**:
   - Recognizes text in images with consistent fonts using preloaded character templates.
   - Efficient for recognizing numbers and characters from digital displays.

2. **CNN-Based OCR**:
   - Uses PaddleOCR to detect and recognize complex or distorted text, such as CAPTCHAs.
   - Provides a fallback mechanism for cases where template matching fails.

3. **Preprocessing**:
   - Converts images to grayscale and applies thresholding to improve recognition accuracy.

---

## Prerequisites
1. Python 3.7 or higher.
2. Install required libraries:
   ```bash
   pip install paddleocr
   pip install paddlepaddle==2.4.2
   pip install opencv-python
   ```

---

## Directory Structure
```
.
├── templates/              # Folder containing character templates for template matching
│   ├── 0.jpg               # Template image for character '0'
│   ├── 1.jpg               # Template image for character '1'
│   ├── 2.jpg               # Template image for character '2'
│   └── ...                 # Add templates for other characters
├── display (3).jpg         # Example image (digital display)
├── display (2).jpg         # Example image (digital display)
├── capchas (1).jpeg        # Example image (CAPTCHA)
└── script.py               # Main OCR script
```

---

## Code Explanation

### 1. **Preprocessing**
Prepares the image for OCR by converting it to grayscale and applying binary thresholding to enhance text regions.
```python
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return image, gray, thresh
```

### 2. **Template Matching**
Compares the input image to predefined character templates and recognizes predictable fonts like those on digital displays.
```python
def template_matching(gray_image, templates, threshold=0.8):
    recognized_text = ""
    for char, template_path in templates.items():
        template = cv2.imread(template_path, 0)
        if template is None:
            print(f"Error: Could not load template from {template_path}")
            continue
        if template.shape[0] > gray_image.shape[0] or template.shape[1] > gray_image.shape[1]:
            h, w = template.shape
            scale_factor = min(gray_image.shape[0] / h, gray_image.shape[1] / w)
            new_size = (int(w * scale_factor), int(h * scale_factor))
            template = cv2.resize(template, new_size)
        result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val > threshold:
            recognized_text += char
    return recognized_text
```

### 3. **CNN-Based OCR**
Uses PaddleOCR for text detection and recognition in images where template matching fails.
```python
def cnn_ocr(image_path):
    result = ocr.ocr(image_path, det=True, rec=True)
    detected_text = ""
    for line in result[0]:
        detected_text += line[1][0] + " "
    return detected_text.strip()
```

### 4. **Hybrid OCR**
Combines both methods, starting with template matching and falling back to CNN OCR when template matching produces no results.
```python
def hybrid_ocr(image_path, templates):
    _, gray, _ = preprocess_image(image_path)
    tm_result = template_matching(gray, templates)
    if not tm_result:
        tm_result = cnn_ocr(image_path)
    return tm_result
```

### 5. **Main Execution**
Processes a list of input images and prints the recognized text.
```python
templates = {
    "0": "templates/0.jpg",
    "1": "templates/1.jpg",
    "2": "templates/2.jpg",
    # Add templates for other characters
}

image_paths = [
    "display (3).jpg",
    "display (2).jpg",
    "capchas (1).jpeg",
]

for path in image_paths:
    print(f"Text from {path}: {hybrid_ocr(path, templates)}")
```

---

## Usage

1. **Prepare Templates**:
   - Place character templates in the `templates/` folder.
   - Ensure file names match the corresponding character (e.g., `0.jpg` for '0').

2. **Run the Script**:
   - Execute the script to recognize text from images.
   ```bash
   python script.py
   ```

3. **Output**:
   - Recognized text for each image will be printed to the console.

---

## Example Output
For the provided example images:
```
Text from display (3).jpg: 23.29
Text from display (2).jpg: 66.8
Text from capchas (1).jpeg: N6R6VR
```

---

## Enhancements
1. **Expand Template Library**:
   - Add templates for a wider range of characters and fonts.

2. **Dynamic Confidence Threshold**:
   - Adjust the `threshold` in template matching dynamically based on image quality.

3. **Integrate Deep Learning**:
   - Train a lightweight CNN model for cases requiring custom OCR capabilities.

4. **Error Handling**:
   - Improve error messages and add logging for debugging.

---

## Dependencies
1. [OpenCV](https://opencv.org/) for image processing.
2. [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for CNN-based OCR.
3. [PaddlePaddle](https://www.paddlepaddle.org.cn/) as a backend for PaddleOCR.

--- 
