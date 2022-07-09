import cv2
import numpy
from os import listdir
from matplotlib import pyplot as plt

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

cleaned_img_gray = cv2.imread(f"{dirImagesList[0]}-cleaned.png", cv2.IMREAD_GRAYSCALE)
"""
GLOBAL THRESHOLDING EXPERIMENT -> DOENS'T WORK
cleaned_img_gray = cv2.cvtColor(cleaned_img_gray, cv2.COLOR_BGR2GRAY)
print(cleaned_img_gray)
ret, mask = cv2.threshold(cleaned_img_gray, 54000, 65535, cv2.THRESH_)
cv2.imshow("Binary image", mask)
cv2.waitKey(0)
"""

cleaned_img = cv2.medianBlur(cleaned_img_gray,5)
ret,th1 = cv2.threshold(cleaned_img,127,256,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(cleaned_img,256,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(cleaned_img,256,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [cleaned_img, th1, th2, th3]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

