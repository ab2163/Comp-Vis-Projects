# Run with exec(open('DogsVsCats.py').read())

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
 
# Define CNN model
def define_model():
	model = Sequential()
	random_uniform = RandomUniform(minval=-0.5, maxval=0.5)
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=random_uniform, padding='same', input_shape=(200, 200, 3)))
	
	# VGG16 has TWO convolution layers before pooling
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=random_uniform, padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer=random_uniform, padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	
	# VGG16 has TWO hidden layers before the output layer
	model.add(Dense(128, activation='relu', kernel_initializer=random_uniform))
	model.add(Dense(1, activation='sigmoid'))
	
	# Use Adam optimiser
	opt = Adam(learning_rate=0.001)
	
	# Use MSE loss function
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model
 
# Diagnostic learning curves
def diagnostic_plots(history):
	
	# Loss function plot
	pyplot.subplot(2, 1, 1)
	pyplot.title('Loss Function vs Epochs')
	pyplot.plot(history.history['loss'], color='blue', label='Training Set')
	pyplot.plot(history.history['val_loss'], color='orange', label='Test Set')
	
	# Plot accuracy
	pyplot.subplot(2, 1, 2)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='Training Set')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='Test Set')

	# Save plots to file
	pyplot.savefig('Diagnostic_Plots_Run3.png')
	pyplot.close()
 
def train_model():
	model = define_model()

	# Create data generators
	training_gen = ImageDataGenerator(rescale=1.0/255.0,
		width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
	testing_gen = ImageDataGenerator(rescale=1.0/255.0)

	# Create iterators
	training_iterator = training_gen.flow_from_directory('content/Dogs_Vs_Cats/Train',
		class_mode='binary', batch_size=64, target_size=(200, 200))
	testing_iterator = testing_gen.flow_from_directory('content/Dogs_Vs_Cats/Test/',
		class_mode='binary', batch_size=64, target_size=(200, 200))

	checkpoint = ModelCheckpoint('CNNDogsVsCats.hdf5', monitor='val_accuracy', verbose=1,
    	save_best_only=False, mode='auto', save_freq='epoch')

	# Fit model
	history = model.fit_generator(training_iterator, steps_per_epoch=len(training_iterator),
		validation_data=testing_iterator, validation_steps=len(testing_iterator), epochs=100, verbose=2, callbacks=[checkpoint])

	# Evaluate model
	scores = model.evaluate_generator(testing_iterator, steps=len(testing_iterator), verbose=2)

	# Plot accuracy for test set
	print('> %.3f' % (scores[1] * 100.0))

	# Plot diagnostic curves
	diagnostic_plots(history)
 
# Line to run code
train_model()