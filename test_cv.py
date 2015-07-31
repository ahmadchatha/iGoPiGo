# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import json
import blescan
import sys
# this is to make opencv available anywhere
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
from gopigo import *
import numpy as np
from sklearn.externals import joblib
import bluetooth._bluetooth as bluez
from firebase import Firebase

# get a reference to firebace
f = Firebase('https://gopigo.firebaseio.com/beacon')

# this is bluetooth setup
dev_id = 0
try:
        sock = bluez.hci_open_dev(dev_id)
        print "ble thread started"

except:
        print "error accessing bluetooth device..."
        sys.exit(1)

blescan.hci_le_set_scan_parameters(sock)
blescan.hci_enable_le_scan(sock)

# load the model
dbn = joblib.load('dbn.pkl')
# get the target
target = int(sys.args[1]) 

# this function will take the picture, process it and give prediction
def get_pred(camera, rawCapture):
	 
	# allow the camera to warmup
	time.sleep(0.1)
	 
	# grab an image from the camera
	camera.capture(rawCapture, format="bgr")
	image = rawCapture.array
	
	# rotate 180
	(h, w) = image.shape[:2]
	center = (w / 2, h / 2)
	M = cv2.getRotationMatrix2D(center, 180, 1.0)
	rotated = cv2.warpAffine(image, M, (w, h))
	
	# crop
	(h, w) = image.shape[:2]
	cropped = rotated[(h/2.5):h, 0:w]
	r = 280.0 / cropped.shape[1]
	dim = (196,98)
	# resize
	resized = cv2.resize(cropped, dim, interpolation = cv2.INTER_AREA)
	# canny edge detection
	img = cv2.Canny(resized,100,200)
	s = img.shape[0] * img.shape[1]
	# flatten the image
	img_wide = img.reshape(1, s)
	img = np.array(img)
	# make a prediction using the model
	pred = dbn.predict(np.atleast_2d(img))
	return pred

# this function returns the closest beacon id
def get_beacon():
	final = 0
	signals = {}
    returnedList = blescan.parse_events(sock, 10)
    for beacon in returnedList:
            #print beacon
            parameters = beacon.split(",")
            if(int(parameters[2]) == 5000):
                    signals[parameters[3]] = parameters[5]

    final = (min(signals, key=signals.get))

# get the camera reference
camera = PiCamera()
rawCapture = PiRGBArray(camera)
# get the current beacon
current = int(get_beacon())
# main loop
while current != target:
	rawCapture.truncate(0)
	a = get_pred(camera,rawCapture)
	if a==1:
		fwd()
		time.sleep(2)
		stop()
	# here we simplify the soft-left and hard-left to be just left		
	elif a==3 or a==4:
		left()
		time.sleep(.2)
		stop()
	# here we simplify the soft-right and hard-right to be just right	
	elif a==5 or a==6:
		right()
		time.sleep(.2)
		stop()
	else:
		print "wrong prediction"
	current = int(get_beacon())
	# send the beacon id to firebase
    result = f.update({'value': current})

print ('YOU HAVE ARRIVED AT YOUR DESTINATION')
