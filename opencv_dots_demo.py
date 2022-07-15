import cv2
import numpy
from os import listdir
from matplotlib import pyplot as plt

"""
-----------------------------------------------
ALGORITHM
- Load images as PIL.Image
- Iterate over every Image
  - Clean Image
	 - Remove Black Border
      - Global Thersholding with Binary Threshold
	 - Recognize Cirlces and Cut out everything outside of them
	 - Calculate mean pixel value within each circle
-----------------------------------------------
"""

dirImagesList = [imageName for imageName in listdir() if ".tiff" in imageName]
print(dirImagesList)


for i in range(0,len(dirImagesList)):
  #img_unchanged = cv2.imread(dirImagesList[0], cv2.IMREAD_ANYDEPTH)
  img_unchanged = cv2.imread(dirImagesList[i], cv2.IMREAD_UNCHANGED)
  cv2.imshow("img_unchanged", img_unchanged)
  h,w= img_unchanged.shape
  print(f"Original Height and Width: {h}x{w}, dtype: {img_unchanged.dtype}")

  img_grayscale = cv2.imread(dirImagesList[i], cv2.IMREAD_GRAYSCALE)
  cv2.imshow("img_grayscale", img_grayscale)
  print(img_grayscale.dtype)

  """
  BINARY THRESHOLD ALGORITHM:
  if (src(x,y) > thresh):
    dst(x,y) = maxValue
  else:
    dst(x,y) = 0
  """
  print(img_unchanged)
  threshold = 5000
  maxValue = 65535
  th, result = cv2.threshold(img_unchanged, threshold, maxValue, cv2.THRESH_BINARY)

  cv2.imshow("img_thresholded", result)

  """
  ALPHA BLEND MASK

  image_blended = cv2.bitwise_and(img_unchanged, result)
  bgr = image_blended[:,:3]
  gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
  bgr = icv2.cvtColor(gray ,cv2.COLOR_GRAY2BGR)
  alpha = src[:,3]
  image_blended = np.dstack([bgr, alpha])
  """

  img_3gray = cv2.merge((img_unchanged, img_unchanged, img_unchanged, result))
  cv2.imshow("im_3gray", img_3gray)

  cv2.imwrite(f"{dirImagesList[0]}-cleaned.tiff", img_3gray)

  cv2.waitKey(0)
  cv2.destroyAllWindows()

  cleaned_img_gray = cv2.imread(f"{dirImagesList[0]}-cleaned.tiff", cv2.IMREAD_GRAYSCALE)
  """
  GLOBAL THRESHOLDING EXPERIMENT -> DOENS'T WORK AT THIS STAGE
  LOOK FOR OTHER METHODS: Adaptive Mean Thresholding, Adaptive Gaussian Thresholding
  """

  cleaned_img = cv2.medianBlur(cleaned_img_gray,5)
  ret,th1 = cv2.threshold(cleaned_img,127,256,cv2.THRESH_BINARY)
  th2 = cv2.adaptiveThreshold(cleaned_img,256,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
  th3 = cv2.adaptiveThreshold(cleaned_img,256,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
  titles = ['Original Image', 'Global Thresholding (v = 127)','Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
  images = [cleaned_img, th1, th2, th3]
  for j in range(4):
    plt.subplot(2,2,j+1),plt.imshow(images[j],'gray')
    plt.title(titles[j])
    plt.xticks([]),plt.yticks([])
  plt.show()
  if (i > 0):
    exit(0) #Remove this if clause to let this script 

