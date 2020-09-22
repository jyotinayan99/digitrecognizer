!pip install tensorflow keras numpy mnist matplotlib

import numpy as np
import mnist  #Get data set from
import matplotlib.pyplot as plt #Graph

from keras.models import Sequential #ANN architecture
from keras.layers import Dense  #The layers in the ANN
from keras.utils import to_categorical


train_images = mnist.train_images() #training data images
train_labels = mnist.train_labels() #training data labels
test_images = mnist.test_images() #training data images
test_labels = mnist.test_labels()


#Normalize the images. Normalize the pixel values from [0,255] to [0 to 1]
train_images = (train_images/255)
test_images = (test_images/255)
#Flatten the images from 28x28 to 784px
train_images = train_images.reshape((-1,784))
test_images = test_images.reshape((-1,784))
#Print the shape
print(train_images.shape)
print(test_images.shape)


#Build the model
# 3 layers, 2 layers with 64 neurons and the relu function
# 1 layer with 10 neurons and softmax function
model = Sequential()
model.add( Dense(64, activation='relu', input_dim=784))
model.add( Dense(64, activation='relu'))
model.add( Dense(10, activation='softmax'))


#Compile the model
#The loss function measures how well the model did on training and then tries to improve the answer
model.compile(
    optimizer='adam',
      loss = 'categorical_crossentropy', #(classes that are greater than 2)
      metrics = ['accuracy']
)


#Train the model
model.fit(
    train_images,
      to_categorical(train_labels), #Ex. 2 it expects 10 dimensional vector 
      epochs = 5, #The number of iterations over the entire dataset to trainon
      batch_size = 32 #The number of samples per gradient
)


#Evaluate the model
model.evaluate(
    test_images,
      to_categorical(test_labels)
)

#model.save_weights('model.h5')

#predict on the first 5 test images

predictions = model.predict(test_images[:5])
# print(predictions)
# print our models prediction
print(np.argmax(predictions, axis =1))
print(test_labels[:5])


for i in range(0,5):
  first_image = test_images[i]
  first_image = np.array(first_image, dtype='float')
  pixels = first_image.reshape((28, 28))
  plt.imshow(pixels, cmap='gray')
  plt.show()

