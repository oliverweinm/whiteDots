from PIL import Image, ImageFilter
from os import listdir
import numpy

"""
-----------------------------------------------
ALGORITHM
- Load images as PIL.Image
- Clean Image
	- Remove Black Border
	- Recognize Cirlces and Cut out everything outside of them
	- Calculate mean pixel value within each circle
-----------------------------------------------

The images are loaded by PIL with mode "I", the arrays are made up of 32-bit signed integer values (-2 Million to 2 Million)

"""





#numpy.set_printoptions(threshold=sys.maxsize)
#numpy.set_printoptions(threshold=numpy.inf)

#Collect all filenames of present .tiff images
dirImagesList = [imageName for imageName in listdir() if ".tiff" in imageName]

print(dirImagesList)


"""
for image in dirImagesList:
	imageCanvas = Image.open(image)
	imageCanvas.show()
	imageArray = numpy.array(imageCanvas)
	print(imageArray)
"""

image = Image.open(dirImagesList[0], formats=["TIFF"])
print(image.info)
#print(image.tag_v2)
image.show()
imageArray = numpy.array(image)

print(image.getbands()) #Image file is of mode 32-bit signed integer pixels
print(imageArray)


sharp_image = image.convert("RGB")
sharp_image = sharp_image.filter(filter=ImageFilter.SHARPEN)
sharp_image.show()
print(sharp_image.getbands())

"""
bw_1_im = image.convert("1")
bw_1_im.show()

bw_8_im = image.convert("L")
bw_8_im.show()

print(numpy.array(bw_8_im))


for x in range(0,50):
	for y in range(0,50):
		imageArray[x,y] = 100000
"""

for x in range(0, 1080):
	for y in range(0, 1080):
		if imageArray[x,y] < 4250: #This an extreme mask that tries to cut out everything except for the test plate. 300-400 is the value where significant areas of background start to get masked:
			imageArray[x,y] = 180000

im = Image.fromarray(imageArray)
im.show()
print(imageArray)


