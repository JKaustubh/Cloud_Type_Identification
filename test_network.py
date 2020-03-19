# USAGE
# python test_network.py --model santa_not_santa.model --image images/examples/santa_01.png

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
orig = image.copy()

# pre-process the image for classification
image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])

# classify the input image
#(notHorse, Horse) = model.predict(image)[0]

# build the label
#label = "Horse" if Horse > notHorse else "Not Horse"
#proba = Horse if Horse > notHorse else notHorse
#label = "{}: {:.2f}%".format(label, proba * 100)

# classify the input image
#(notFlower, Flower) = model.predict(image)[0]

# build the label
#label = "Flower" if Flower > notFlower else "Not Flower"
#proba = Flower if Flower > notFlower else notFlower
#label = "{}: {:.2f}%".format(label, proba * 100)

(notGold, Gold) = model.predict(image)[0]

#build the label
label = "Gold Fish" if Gold > notGold else "Not Gold Fish"
proba = Gold if Gold > notGold else notGold
label = "{}: {:.2f}%".format(label, proba * 100)

#(notSanta, Santa) = model.predict(image)[0]

#build the label
#label = "Santa" if Santa > notSanta else "Not Santa"
#proba = Santa if Santa > notSanta else notSanta
#label = "{}: {:.2f}%".format(label, proba * 100)

#(notUniform, Uniform) = model.predict(image)[0]

#build the label
#label = "Uniform" if Uniform > notUniform else "Not Uniform"
#proba = Uniform if Uniform > notUniform else notUniform
#label = "{}: {:.2f}%".format(label, proba * 100)

#(notCirrus, Cirrus) = model.predict(image)[0]

#build the label
#label = "Cirrus Cloud" if Cirrus > notCirrus else "Not Cirrus"
#proba = Cirrus if Cirrus > notCirrus else notCirrus
#label = "{}: {:.2f}%".format(label, proba * 100)

# draw the label on the image
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)

