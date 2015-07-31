# This file trains the network 
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn import datasets
from nolearn.dbn import DBN
import numpy as np
from sklearn.metrics import zero_one_loss
import cv2
import glob, os
from sklearn.externals import joblib


def img_to_matrix(filename, verbose=False):
	image = cv2.imread(filename,0)
	dim = (196,98)
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	edges = cv2.Canny(resized,100,200)
	return edges

def flatten_image(img):
	s = img.shape[0] * img.shape[1]
	img_wide = img.reshape(1, s)
	return img_wide[0]

data = []
target = []
# the directory where the images are
os.chdir("/Users/achatha/Documents/NLP/images/training")
# getting all the .png images
# Initially during training, we had 6 labels (1- forward, 2- longer forward, 3- soft left, 4- hard left, 5-soft right, 6-hard right)
# but then in order to get better predictions i made 1=2 sice they are the same forward command.
for path in glob.glob('*.png'):
	dirname, filename = os.path.split(path)
	label = (int(filename.split('_')[1]))
	if label==2:
		label = 1
	target.append(label)
	img = img_to_matrix(filename)
	img = flatten_image(img)
	data.append(img)

data = np.array(data)
target = np.array(target)

# here we do a 75/25 split and also normalize the pixel values by dividing by 255
(trainX, testX, trainY, testY) = train_test_split(
	data / 255.0, target.astype("int0"), test_size = 0.25)

# i also tried to do Principal component analysis but with out any imporvements in the test error
'''
std_scale = preprocessing.StandardScaler().fit(train_x)
X_train_std = std_scale.transform(train_x)
X_test_std = std_scale.transform(test_x)

pca_std = PCA(n_components=6).fit(X_train_std)
trainX = pca_std.transform(X_train_std)
testX = pca_std.transform(X_test_std)

'''
# here we train the network by setting the number of input nodes to be the dimension of processed image,
# hidden layer has 2000 nodes and output layers has 5 corresponding to the 5 labels
dbn = DBN(
	[trainX.shape[1],2000, 5],
	learn_rates = 0.3,
	learn_rate_decays = 0.9,
	epochs = 6,
	verbose = 1)
dbn.fit(trainX, trainY)

# we dump the model so that we can run it and get predictions on the PI
joblib.dump(dbn, 'dbn.pkl')

# compute the predictions for the test data and show a classification
# report
preds = dbn.predict(testX)
error = zero_one_loss(testY, preds)
print("error_rate: " + str(error))
print classification_report(testY, preds)

# randomly select a few of the test instances
# just to see if things are working and show to someone.
for i in np.random.choice(np.arange(0, len(testY)), size = (10,)):
	# classify the digit
	pred = dbn.predict(np.atleast_2d(testX[i]))
 
	# reshape the feature vector to be a 196x98 pixel image
	image = (testX[i] * 255).reshape(98, 196 )
 
	# show the image and prediction
	print "Actual digit is {0}, predicted {1}".format(testY[i], pred[0])
	cv2.imshow("Digit", image)
	cv2.waitKey(0)
