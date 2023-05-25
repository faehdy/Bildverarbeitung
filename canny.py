import numpy as np
from scipy.ndimage import gaussian_filter,convolve
import matplotlib.pyplot as plt
from scipy import where
import cv2

DPI = 120


def non_max_suppression(magnitude, direction):
    M, N = magnitude.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = direction * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255

                #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = magnitude[i, j+1]
                    r = magnitude[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = magnitude[i+1, j-1]
                    r = magnitude[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = magnitude[i+1, j]
                    r = magnitude[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = magnitude[i-1, j-1]
                    r = magnitude[i+1, j+1]

                if (magnitude[i,j] >= q) and (magnitude[i,j] >= r):
                    Z[i,j] = magnitude[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass

    return Z


def canny_edge_detector(input_img, sigma = 1):
    input_img=cv2.imread('inputs/grossmuenster.png', 0).astype('float')
    plt.figure(figsize=(5, 5),dpi=DPI)
    plt.xticks([]), plt.yticks([])
    plt.imshow(input_img, cmap = 'gray')
    plt.title('Input image')


    blurred_img = gaussian_filter(input_img, sigma)

    plt.figure(figsize=(12, 12),dpi=DPI)
    plt.subplot(121)
    plt.xticks([]), plt.yticks([])
    plt.imshow(input_img, cmap = 'gray')
    plt.title('Input image')

    plt.subplot(122)
    plt.xticks([]), plt.yticks([])
    plt.imshow(blurred_img, cmap=plt.cm.gray)
    plt.title('Blurred image')



    sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    I_x = convolve(blurred_img, sobel_kernel)
    I_y = convolve(blurred_img, sobel_kernel.T)


    magnitude = np.sqrt(I_x**2 + I_y**2)
    direction = np.arctan2(I_y, I_x)



    thresh= 0.1 * magnitude.max()
    thresholdEdges = (magnitude > thresh)
    magnitude = np.where(thresholdEdges, magnitude, 0)



    edges = non_max_suppression(magnitude, direction)
    edges = np.where(edges > 0, 255, 0)

