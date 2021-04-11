
import argparse 
import pickle 

import cv2

import numpy as np 
import operator as op 
import itertools as it, functools as ft 


import mediapipe as mp 

from scipy.spatial.distance import euclidean as measure 

def get_nodes_from_landmark(landmark, scaler):
	nodes = [ [pnt.x, pnt.y] for pnt in landmark.landmark]
	return (np.array(nodes) * scaler).astype('int32') 

def create_screen(screen_name, screen_w, screen_h, position=None):
	cv2.namedWindow(screen_name, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(screen_name, screen_w, screen_h)
	if position is not None: 
		cv2.moveWindow(screen_name, *position)

def get_hand_bounder(hand_nodes):
	return cv2.boundingRect(hand_nodes)


def create_graph(nodes):
	nb_nodes = len(nodes)
	pairs = list(it.product(nodes, nodes))
	weighted_edges = [ measure(fst, snd) for fst, snd in pairs ]
	adjacency_matrix =  np.array(weighted_edges)
	return np.reshape(adjacency_matrix, (nb_nodes, nb_nodes))

def to_image(matrix, W, H):
	integer_matrix = (matrix * 255).astype('uint8')
	resized_matrix = cv2.resize(integer_matrix, (W, H), interpolation=cv2.INTER_CUBIC)
	return resized_matrix


def grab_process():
	try:
		mp_builder = mp.solutions.hands 
		mp_drawing = mp.solutions.drawing_utils 

		mp_builder_config = {
			'max_num_hands': 2, 
			'min_tracking_confidence': 0.7, 
			'min_detection_confidence': 0.7
		}

		drawing_spec_0 = mp_drawing.DrawingSpec(color=(255, 0, 255), circle_radius=5, thickness=3)
		drawing_spec_1 = mp_drawing.DrawingSpec(color=(  0, 0, 255), thickness=3)

		with mp_builder.Hands(**mp_builder_config) as detector:
			print('mediapipe successfull ...!')
			W, H = 640, 480
			scaler = np.array([W, H])

			screen_0 = 'main-0000'
			screen_1 = 'main-0001'
			create_screen(screen_0, W, H, ( 100, 100))
			create_screen(screen_1, W, H, (1000, 100))
			
			sigma = 'lnr'
			center = np.array([W, H]) // 2 

			features_accumulator = []
			capture = cv2.VideoCapture(0)
			keep_grabing = True 
			while keep_grabing:
				key_code = cv2.waitKey(25) & 0xFF 
				keep_grabing = key_code != 27 
				capture_status, bgr_frame = capture.read()
				if capture_status:
					rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
					response = detector.process(rgb_frame)
					output = response.multi_hand_landmarks
					if output:
						first_hand = output[0]
						mp_drawing.draw_landmarks(
							bgr_frame, 
							first_hand, 
							mp_builder.HAND_CONNECTIONS, 
							drawing_spec_0, 
							drawing_spec_1
						)
						nodes = get_nodes_from_landmark(first_hand, scaler)
						adjacency_matrix = create_graph(np.vstack([nodes, center[None, :]]))

						adjacency_matrix /= np.max(adjacency_matrix)
						image_matrix = to_image(adjacency_matrix, W, H)
						cv2.imshow(screen_1, image_matrix)
							
						if chr(key_code) in sigma: 
							print(key_code, chr(key_code))
							features_accumulator.append((np.ravel(adjacency_matrix), key_code)) 

						x, y, w, h = get_hand_bounder(nodes)
						cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), (0, 255, 255), 3)
						
						
					cv2.imshow(screen_0, bgr_frame)
			# end LOOP ...!

			with open('dump/features.pkl', 'wb') as file_pointer:
				pickle.dump(features_accumulator, file_pointer)

		# end context manager ...!

	except KeyboardInterrupt as e: 
		print('[keyboard] interrupt was catched', e)
	except Exception as e:
		print('[exception] was catched', e)

if __name__ == '__main__':
	print(' ... [processing] ... ')
	grab_process()
