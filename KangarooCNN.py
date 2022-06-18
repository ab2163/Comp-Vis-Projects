# Run with exec(open('KangarooCNN.py').read())

import sys
import tensorflow as tf
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from keras.initializers import RandomUniform
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg16 import VGG16
 
# Define CNN model
def define_model():

	vggmodel = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
	vggmodel.trainable = False

	flatten_layer = Flatten()
	dense_layer_1 = Dense(400, activation='relu')
	dense_layer_2 = Dense(100, activation='relu')
	dense_layer_3 = Dense(25, activation='relu')
	prediction_layer = Dense(2, activation='softmax')

	model = Sequential([vggmodel, flatten_layer, dense_layer_1, dense_layer_2, dense_layer_3, prediction_layer])
	
	# Use Adam optimiser
	opt = Adam(learning_rate=0.001)
	
	# Use cross entropy loss function
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def train_model():
	model = define_model()

	# Create data generators
	training_gen = ImageDataGenerator(rescale=1.0/255.0,
		width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
	testing_gen = ImageDataGenerator(rescale=1.0/255.0)

	# Create iterators
	training_iterator = training_gen.flow_from_directory('Kangaroo/Train',
		class_mode='categorical', batch_size=64, target_size=(224, 224))
	testing_iterator = testing_gen.flow_from_directory('Kangaroo/Test/',
		class_mode='categorical', batch_size=64, target_size=(224, 224))

	checkpoint = ModelCheckpoint('CNNKangarooVGG16.hdf5', monitor='val_accuracy', verbose=1,
    	save_best_only=True, mode='auto', save_freq='epoch')

	# Fit model
	history = model.fit_generator(training_iterator, steps_per_epoch=len(training_iterator),
		validation_data=testing_iterator, validation_steps=len(testing_iterator), epochs=500, verbose=2, callbacks=[checkpoint])
 
# Line to run code
train_model()