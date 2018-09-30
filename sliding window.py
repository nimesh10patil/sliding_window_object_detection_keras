import time
import cv2
from keras.models import model_from_json # loading the model which was previously saved
import numpy as np
#from scikit-image.measure import structural_similarity as ssim
import os

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#model

classifier = Sequential()
classifier.add(Convolution2D(64, 3, 3, input_shape = (128, 128, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(64, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


classifier.add(Convolution2D(128, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())


# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.summary()
#load weights

classifier.load_weights('cats_dogs.h5')


#preproess
img = cv2.imread('cat.jpg', 1)
resized = cv2.resize(img,(128,128))/255 # Pre-processing the image and normalize
wind_row, wind_col = 45,45 # dimensions of the image
img_rows, img_cols = 128,128


#generating the sliding window
def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


# predicting the image
def prd(window_image):
    return classifier.predict(window_image)


for (x, y, window) in sliding_window(resized, 15, (45, 45)):
    print("hiiii")
    if window.shape[0] != wind_row or window.shape[1] != wind_col:
       print("continues")
       continue
    clone = resized.copy()
    cv2.rectangle(clone, (x, y), (x + wind_row, y + wind_col), (0, 255, 0), 2)

    t_img = resized[y:y + wind_row, x:x + wind_col]  # the image which has to be predicted
    print("t_img",t_img.shape)
    test_img=cv2.resize(t_img,(128,128))
    test_img = np.expand_dims(test_img, axis=0)
                                # expanding the dimensions of the image to meet the dimensions of the trained model
    print("test_img",test_img.shape)

    prediction = prd(test_img)  # predict the image
    classes = prediction[0]
    if classes < 0.5 :
        print("cat")
    else:
        print("dog")

    cv2.imshow("sliding_window", resized[y:y + wind_row, x:x + wind_col])
    cv2.imshow("Window", clone)
    cv2.waitKey(1)
    time.sleep(1.5)


print("up")
cv2.waitKey(0)
cv2.destroyAllWindows()
