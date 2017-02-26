import cv2
import numpy as np
from PIL import Image

img = cv2.imread('images/front.bmp',0)
mask = cv2.imread('images/mask_front.bmp', 0)


dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

cv2.imwrite('out.png', dst)

