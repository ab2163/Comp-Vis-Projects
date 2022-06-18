# Run with exec(open('DetectKangaroo.py').read())

import cv2
import numpy as np
from tensorflow import keras
from numpy import asarray 

# Reload model
model = keras.models.load_model('CNNKangarooVGG16.hdf5')

def perform_selective_search(input_image):

	ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
	ss.setBaseImage(input_image)

	# Algorithm will not run unless speed is specified
	ss.switchToSelectiveSearchFast()
	
	output_rectangles = ss.process()
	return output_rectangles

def get_proposed_images(input_image, proposed_rectangles):

	# Arrays of regions to output
	region_arrays = list()

	for count, rect in enumerate(proposed_rectangles):
		if count < 2000:
			x, y, w, h = rect
			
			image_region = input_image[y:y+h, x:x+w]

			# Inter Linear chosen since images most likely to be enlarged
			image_region = cv2.resize(image_region, (224,224), interpolation = cv2.INTER_LINEAR)

			# Rescale images to match training environment
			image_region = image_region.astype('float32')
			image_region = (1.0/255.0)*image_region

			# Add to list
			region_arrays.append(image_region)

		else:
			break

	# Convert to NumPy arrays
	region_arrays = asarray(region_arrays)
	return region_arrays

def get_IOU(rectA, rectB):

	x1_A, y1_A, w_A, h_A = rectA
	x2_A = x1_A + w_A
	y2_A = y1_A + h_A
	x1_B, y1_B, w_B, h_B = rectB
	x2_B = x1_B + w_B
	y2_B = y1_B + h_B

	x_left = max(x1_A, x1_B)
	y_bottom = max(y1_A, y1_B)
	x_right = min(x2_A, x2_B)
	y_top = min(y2_A, y2_B)

	# If rectangles do not intersect, return 0
	if x_right < x_left or y_bottom > y_top:
		return 0.0, 0.0, 0.0

	intersection_area = (x_right - x_left) * (y_top - y_bottom)

	IOU = intersection_area / float(w_A*h_A + w_B*h_B - intersection_area)
	IOA = intersection_area/ float(w_A*h_A)
	IOB = intersection_area/ float(w_B*h_B)

	return IOU, IOA, IOB

def non_maximum_suppression(proposed_rectangles, rectangle_scores, cutoff):

	proposed_rectangles = proposed_rectangles[:2000]
	
	# Initially assume all rectangles are suitable candidates
	rect_selection = np.ones(2000);

	# Deselect if score < cutoff
	# "count" will iterate from 0 to 1999
	for count, score in enumerate(rectangle_scores[:,1]):
		if score < cutoff:
			rect_selection[count] = 0

	for count1, rect1 in enumerate(proposed_rectangles):

		if rect_selection[count1] == 1:
			for i, rect2 in enumerate(proposed_rectangles[count1+1:2000:1]):

				count2 = i + (count1+1)

				if rect_selection[count2] == 1:

					rect_IOU, int_rect1, int_rect2 = get_IOU(rect1, rect2)

					# Intersection threshold passed, the lesser region will be deselected
					if rect_IOU > 0.5 or int_rect1 > 0.33 or int_rect2 > 0.33:

						# If rectangle 1 is high scoring, deselect rectangle 2
						if rectangle_scores[count1, 1] > rectangle_scores[count2, 1]:
							rect_selection[count2] = 0

						# Otherwise deselect rectangle 1 and move on to next rectangle
						else:
							rect_selection[count1] = 0
							break
					
	selected_rects = list()
	selected_scores = list()

	# Create lists of selected rectangles and scores
	for count, score in enumerate(rectangle_scores[:,1]):

		if rect_selection[count] == 1:
			selected_rects.append(proposed_rectangles[count])
			selected_scores.append(rectangle_scores[count, 1])
			print(rectangle_scores[count, 1])

	# Convert lists to NumPy arrays
	selected_rects = asarray(selected_rects)
	selected_scores = asarray(selected_scores)

	return selected_rects, selected_scores

def visualise_selected_regions(input_image, input_rects, input_scores, output_filename):

	output_image = input_image.copy()

	for count, rect in enumerate(input_rects):
		x, y, w, h = rect
		color = list(np.random.random(size=3) * 256)
		cv2.rectangle(output_image, (x, y), (x+w, y+h), color, 2, cv2.LINE_AA)

	output_image = cv2.resize(output_image, (2*output_image.shape[1], 2*output_image.shape[0]), interpolation = cv2.INTER_LINEAR)
	#cv2.imshow('Selective Searched Image', output_image)
	cv2.imwrite(output_filename, output_image)

def detect_roo(image_filename, output_filename):
	# Load image
	sample_image = cv2.imread(image_filename)

	# Convert to RBG
	sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

	rects = perform_selective_search(sample_image)
	proposed_regions = get_proposed_images(sample_image, rects)
	region_scores = model.predict(proposed_regions, verbose=1)
	rselect, sselect = non_maximum_suppression(rects, region_scores, 0.9)
	visualise_selected_regions(sample_image, rselect, sselect, output_filename)

detect_roo('roo1.jpg', 'det1.jpg')
detect_roo('roo2.jpeg', 'det2.jpg')
detect_roo('roo3.jpg', 'det3.jpg')
detect_roo('roo4.jpg', 'det4.jpg')
detect_roo('roo5.jpg', 'det5.jpg')
detect_roo('roo6.jpg', 'det6.jpg')
detect_roo('roo7.jpg', 'det7.jpg')
detect_roo('roo8.jpg', 'det8.jpg')
detect_roo('roo9.jpg', 'det9.jpg')
detect_roo('roo10.jpg', 'det10.jpg')
detect_roo('roo11.jpg', 'det11.jpg')
detect_roo('roo12.jpg', 'det12.jpg')

