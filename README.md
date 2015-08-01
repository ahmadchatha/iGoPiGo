# iGoPiGo
Indoor autonomous driving/navigation using ibeacons and GoPiGo.
For more info visit https://gopigo.firebaseapp.com/
## Files
#### dbn.py
This file trains the network. You have mention the directory where the images are for training and what sort of files to read (*.png,*.jpg). You should also modify the network parameters according to your choice. Runnig it will dump model files in the working directory which can be used to make prediction on raspberry pi.
### blescan.py
performs a simple device inquiry, and returns a list of ble advertizements 
### test_cv.py
This takes in the target beacon id as input, then continuously gets the beacon data, takes images, makes navigation predictions until we reach destination beacon. For more details, read the file comments.
