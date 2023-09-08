import cv2
import os

path = os.getcwd()
images = path + "/data/"
imageList = os.listdir(images)

for i in imageList:
	finalPath = images + i
	img = cv2.imread(finalPath) # Read image

	# Defining all the parameters
	t_lower = 500 # Lower Threshold
	t_upper = 600 # Upper threshold
	aperture_size = 5 # Aperture size
	L2Gradient = True # Boolean

	# Applying the Canny Edge filter
	# with Aperture Size and L2Gradient
	edge = cv2.Canny(img, t_lower, t_upper,
					apertureSize = aperture_size,
					L2gradient = L2Gradient )

	# Save the edge image
	outputPath = "edges/" + i
	cv2.imwrite(outputPath, edge)

	cv2.imshow('original', img)
	cv2.imshow('edge', edge)
	cv2.waitKey(0)
	cv2.destroyAllWindows()