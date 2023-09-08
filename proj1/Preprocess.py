import cv2
import numpy as np

def automatic_contrasting(image):
    # Convert to grayscale for better contrast stretch
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Stretch the image to cover full dynamic range
    stretched = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # Now perform histogram equalization for better visual contrast
    equalized = cv2.equalizeHist(stretched)
    
    return equalized


def automatic_white_balance(img):
    # Convert the image to float for calculations
    img_float = np.float32(img) / 255.0
    
    if len(img_float.shape) == 2:  # Grayscale image
        # For grayscale images, simply adjust the illuminant using max or mean
        illuminant = np.max(img_float) if np.max(img_float) > 0 else 1.0  # Avoid division by zero
        
        # Adjusting using the brightest pixel
        scale = 1.0 / illuminant
        white_balanced = img_float * scale
        
    white_balanced = np.clip(white_balanced, 0, 1)
    white_balanced = (white_balanced * 255).astype(np.uint8)
    
    return white_balanced

import os

path = os.getcwd()
newPath = path + "/data"
dir_list = os.listdir(newPath)
print(dir_list)

for i in dir_list:
    finalPath = path + "/data/" + i
    img = cv2.imread(finalPath) # Read image

    img = automatic_contrasting(img)
    img = automatic_white_balance(img)
    outputPath = "filtered/" + i
    cv2.imwrite(outputPath, img)

    cv2.imshow('edge', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()