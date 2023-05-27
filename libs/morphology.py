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

    # Define an initial rectangle that includes the whole image
    rectangle = (0, 0, img.shape[1] - 1, img.shape[0] - 1)

    # Create zero mask
    mask = np.zeros(img.shape[:2], np.uint8)

    # Create two arrays used for GrabCur algorithm
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)


    cv.grabCut(img, mask, rectangle, bgdModel, fgdModel, 16, cv.GC_INIT_WITH_RECT)

    # Create mask, sure and likely backgrounds set  0, otherwise 1
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Multiply image wiht new mask to subtract background
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
    
    # inflate the image 
    kernel = np.ones((3,3),np.uint8)
    dilated = cv.dilate(img, kernel, iterations = 2)

    # erode the image using 
    eroded = cv.erode(dilated, kernel, iterations = 2)

    closed_img = eroded

    return closed_img

def instance_segmentation_util(img):
    """
    Given the closed segmentation image, output the instance segmentation result
    Input:  
        img:        n x m, values are within [0,255]
    Output:
        instance_seg_img:    n x m x 3, different coin instances have different colors
    """
    # Use distance transform and normalization
    img = cv.medianBlur(img,3)
    dist_transform = cv.distanceTransform(img,cv.DIST_L2,3)
    _, sure_fg = cv.threshold(dist_transform,0.32*dist_transform.max(),255,0)

    # Find sure background area
    kernel = np.ones((3,3),np.uint8)
    sure_bg = cv.dilate(img, kernel, iterations = 2)

    # Subtract the sure foreground from the sure background
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)

    # +1 bc the background needs a pixel too
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    # Apply the watershed algorithm
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    markers = cv.watershed(img,markers)

    # Color boundaries in black, 3 Chanels bc of RGB    
    img[markers == -1] = [0,0,0]

    # Create a random colormap
    rand_color = lambda: (int(random.random()*255), int(random.random()*255), int(random.random()*255))
    color_map = [rand_color() for _ in range(np.max(markers)+1)]
    # Create the instance segmentation image as empty image
    instance_seg_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # Iterate over the markers and color the instance segmentation image
    for i in range(markers.shape[0]):
        for j in range(markers.shape[1]):
            instance_seg_img[i,j] = color_map[markers[i,j]]
    instance_seg_img[markers==-1] = [0,0,0]

    plt.imshow(sure_fg, cmap='gray')

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

    #Make the text and the letter binary

    __unnecessary1__, text_binary = cv.threshold(text, 0.6, 1, cv.THRESH_BINARY_INV)
    __unnecessary2__, letter_binary = cv.threshold(letter_not, 0.6, 1, cv.THRESH_BINARY)

    # iterate over every pixel and everywhere where the letter is, erode the text and mark the centroid pixel
    img_err = erode(text_binary, letter_binary)
    # iterate over the eroded image and dilate all of the eroded pixels to get the original letters back.
    img_dil = dilate(img_err, letter_binary)

    return img_dil