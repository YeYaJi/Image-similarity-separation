import argparse

import cv2.cv2
import imutils
import skimage
from skimage.metrics import structural_similarity as ssim



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

imageA = cv2.imread("/home/gsh/PycharmProjects/cloud_dibang/delet_same_img/1")
imageB = cv2.imread("/home/gsh/PycharmProjects/cloud_dibang/delet_same_img/1")

# convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
cv2.namedWindow("12312", cv2.WINDOW_NORMAL)
cv2.imshow("12312",grayA)

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
#(score, diff) = ssim(grayA, grayB, full=True)
# diff = (diff * 255).astype("uint8")


psnr = skimage.metrics.peak_signal_noise_ratio(grayA, grayB)

print("SSIM: {}".format(psnr))
