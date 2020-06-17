# import the necessary packages
from skimage.measure import compare_ssim
from  skimage.metrics import structural_similarity as ssim
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-f", "--first", required=True,
#                 help="first input image")
# ap.add_argument("-s", "--second", required=True,
#                 help="second")
# args = vars(ap.parse_args())

# load the two input images
# imageA = cv2.imread(args["first"])
# imageB = cv2.imread(args["second"])

imageA = cv2.imread("1.png")
imageB = cv2.imread("1.png")

# convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = ssim(grayA, grayB, full=True)
#diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))
cv2.namedWindow("w")
cv2.imshow("w", diff)
cv2.waitKey(0)
cv2.destroyAllWindows()
