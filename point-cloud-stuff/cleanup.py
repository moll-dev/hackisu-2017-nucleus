import cv2
import numpy as np
from PIL import Image

img = cv2.imread('images/test.png',0)
mask = cv2.imread('images/mask.png', 0)

laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

cv2.imwrite('out.png', dst)

