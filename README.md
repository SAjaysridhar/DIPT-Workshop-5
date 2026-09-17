### DIPT-Workshop-5

### License Plate Detection using OpenCV and Haar Cascade Classifier</br>

### Name : AJAY S</br>
### Reg.no : 21224230010</br>
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
img = cv2.imread('car_plate.jpg')

if img is None:
    print("Image not loaded")
else:
    print("Image loaded successfully")
# Function to display image properly in matplotlib
```
```
def display(img):
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12,8))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()
cascade_path = 'haarcascade_licence_plate_rus_16stages.xml'

plate_cascade = cv2.CascadeClassifier(cascade_path)

if plate_cascade.empty():
    print("Error loading cascade classifier")
    print("Current Working Directory:")
    print(os.getcwd())
else:
    print("Cascade classifier loaded successfully!")

def detect_plate(img):
    
    plate_img = img.copy()
    
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    plates = plate_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )
    
    for (x, y, w, h) in plates:
        
        cv2.rectangle(
            plate_img,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            3
        )
    
    return plate_img
```
```
def blur_plate(img):
    
    plate_img = img.copy()
    
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    plates = plate_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )
    
    for (x, y, w, h) in plates:
        
        # Region of Interest (ROI)
        roi = plate_img[y:y+h, x:x+w]
        
        # Blur the ROI
        blurred_roi = cv2.medianBlur(roi, 35)
        
        # Replace original ROI with blurred ROI
        plate_img[y:y+h, x:x+w] = blurred_roi
    
    return plate_img
```
<img width="1230" height="675" alt="646909485-c620e421-16e5-4111-8d31-84e44659f99d" src="https://github.com/user-attachments/assets/957e7bae-5426-49cb-853f-6d883456bd8e" />

<img width="1261" height="629" alt="646909518-7736226b-706e-4b20-8f7a-fd8cd15eab98" src="https://github.com/user-attachments/assets/094970db-7b04-4979-9ce3-f2e6d4ece2da" />

<img width="1340" height="674" alt="646909537-7ce7f44a-624b-4141-918f-99a2bd2bdcac" src="https://github.com/user-attachments/assets/faa37da7-4c04-439b-8a54-7ecee7aa2e17" />


