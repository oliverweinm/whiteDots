import cv2
from os import listdir

"""
-----------------------------------------------
ALGORITHM
- Load images as PIL.Image
- Clean Image
	- Remove Black Border
	- Recognize Cirlces and Cut out everything outside of them
	- Calculate mean pixel value within each circle
-----------------------------------------------
"""

dirImagesList = [imageName for imageName in listdir() if ".tiff" in imageName]
print(dirImagesList)

#img_unchanged = cv2.imread(dirImagesList[0], cv2.IMREAD_ANYDEPTH)
img_unchanged = cv2.imread(dirImagesList[0], cv2.IMREAD_UNCHANGED)
cv2.imshow("img_unchanged", img_unchanged)
h,w= img_unchanged.shape
print(f"Original Height and Width: {h}x{w}, dtype: {img_unchanged.dtype}")

img_grayscale = cv2.imread(dirImagesList[0], cv2.IMREAD_GRAYSCALE)
cv2.imshow("img_grayscale", img_grayscale)
print(img_grayscale.dtype)

"""
BINARY THRESHOLD
if src(x,y) > thresh
  dst(x,y) = maxValue
else
  dst(x,y) = 0
"""
print(img_unchanged)
threshold = 5000
maxValue = 65535
th, result = cv2.threshold(img_unchanged, threshold, maxValue, cv2.THRESH_BINARY)

cv2.imshow("img_thresholded", result)

"""
Alpha Blend Mask

image_blended = cv2.bitwise_and(img_unchanged, result)
bgr = image_blended[:,:3]
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
bgr = icv2.cvtColor(gray ,cv2.COLOR_GRAY2BGR)
alpha = src[:,3]
image_blended = np.dstack([bgr, alpha])
"""

img_3gray = cv2.merge((img_unchanged, img_unchanged, img_unchanged, result))
cv2.imshow("im_3gray", img_3gray)

cv2.imwrite(f"{dirImagesList[0]}-cleaned.png", img_3gray)


cv2.waitKey(0)
cv2.destroyAllWindows()
