import numpy as np
from cv2 import imread,imwrite, dilate, erode
from cv2 import cvtColor, COLOR_BGR2HLS, calcHist
import cv2 as cv
import random
from matplotlib import pyplot as plt
from skimage.measure import label



# --------------------------------- Zusatzaufgabe ---------------------------------------
def segment_util(img):
    """
    Given an input image, output the segmentation result
    Input:  
        img:        n x m x 3, values are within [0,255]
    Output:
        img_seg:    n x m
    """ 
    
    
    
    # # Convert the image to grayscale
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # # Apply Gaussian blur to reduce noise
    # blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # # Apply Otsu's thresholding method to segment the image
    # _, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # # Perform morphological operations to remove small noise - opening
    # # To remove holes we can use closing
    # kernel = np.ones((3,3),np.uint8)
    # opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations = 2)
    # closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations = 2)

    # return closing

def segment_util(img):
    """
    Given an input image, output the segmentation result
    Input:  
        img:        n x m x 3, values are within [0,255]
    Output:
        img_seg:    n x m
    """

    # Define an initial rectangle that includes the whole image
    rectangle = (0, 0, img.shape[1] - 1, img.shape[0] - 1)

    # Create initial mask
    mask = np.zeros(img.shape[:2], np.uint8)

    # Create two arrays used by the GrabCut algorithm internally
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Run GrabCut algorithm
    cv.grabCut(img, mask, rectangle, bgdModel, fgdModel, 20, cv.GC_INIT_WITH_RECT)

    # Create mask where sure and likely backgrounds set to 0, otherwise 1
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Multiply image with new mask to subtract background
    img = img * mask2[:, :, np.newaxis]


    return mask2



def close_hole_util(img):
    """
    Given the segmented image, use morphology techniques to close the holes
    Input:
        img:        n x m, values are within [0,1]
    Output:
        closed_img: n x m
    """
    ## TODO
    closed_img = ...

    return closed_img

def instance_segmentation_util(img):
    """
    Given the closed segmentation image, output the instance segmentation result
    Input:  
        img:        n x m, values are within [0,255]
    Output:
        instance_seg_img:    n x m x 3, different coin instances have different colors
    """
    ## TODO
    instance_seg_img = ...

    return instance_seg_img

def text_recog_util(text, letter_not):
    """
    Given the text and the character, recognise the character in the text
    Input:
        text:           n x m
        letter_not:     a x b
    Output:
        text_er_dil:    n x m
    """
    from scipy.ndimage import binary_erosion as erode
    from scipy.ndimage import binary_dilation as dilate
    ## TODO
    text_er_dil = ...

    return text_er_dil