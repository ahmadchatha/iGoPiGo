from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import PCA
from sklearn import datasets
from nolearn.dbn import DBN
import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob, os
from sklearn.externals import joblib

dbn = joblib.load('dbn.pkl') 

def img_to_matrix(filename, verbose=False):
	image = cv2.imread(filename,0)
	r = 280.0 / image.shape[1]
	#dim = (280, int(image.shape[0] * r))
	dim = (196,98)
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	#sobelx64f = cv2.Sobel(resized,cv2.CV_64F,1,0,ksize=5)
	#abs_sobel64f = np.absolute(sobelx64f)
	#sobel_8u = np.uint8(abs_sobel64f)
	edges = cv2.Canny(resized,100,200)
	#cv2.imshow('image',edges)
	#cv2.waitKey(0)
	return edges

def flatten_image(img):
	s = img.shape[0] * img.shape[1]
	img_wide = img.reshape(1, s)
	return img_wide[0]

data = []
target = []
os.chdir("/Users/achatha/Documents/NLP/images")
for path in glob.glob('*.png'):
	dirname, filename = os.path.split(path)
	target.append(int(filename.split('_')[1]))
	#if count <= 20:
	img = img_to_matrix(filename)
	img = flatten_image(img)
	data.append(img)
	#count = count + 1

data = np.array(data)
target = np.array(target)

(trainX, testX, trainY, testY) = train_test_split(
	data / 255.0, target.astype("int0"), test_size = 0.33)
# randomly select a few of the test instances
for i in np.random.choice(np.arange(0, len(testY)), size = (10,)):
	# classify the digit
	pred = dbn.predict(np.atleast_2d(testX[i]))
 
	# reshape the feature vector to be a 28x28 pixel image, then change
	# the data type to be an unsigned 8-bit integer
	image = (testX[i] * 255).reshape(98, 196 )
 
	# show the image and prediction
	print "Actual digit is {0}, predicted {1}".format(testY[i], pred[0])
	cv2.imshow("Digit", image)
	cv2.waitKey(0)
