import pickle 
import tensorflow as tf 
import numpy as np 

from os import path 

def create_model(input_shape, nb_classes):
	return tf.keras.Sequential([
		tf.keras.Input(shape=input_shape),

		tf.keras.layers.Dense(units=128, activation='relu'), 
		tf.keras.layers.Dense(units=64, activation='relu'),
		tf.keras.layers.Dense(units=nb_classes, activation='softmax') 	
	])

def compile_model(model):
	model.compile(
		optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)

if __name__ == '__main__':
	print(' ... [learning] ... ')

	data_path = 'dump/features.pkl'
	if path.isfile(data_path):
		with open(data_path, 'rb') as file_pointer:
			training_data = pickle.load(file_pointer)
			x_train = [x for x, _ in training_data]
			y_train = [y for _, y in training_data]
			
			x_train = np.vstack(x_train)
			y_train = np.vstack(y_train)

			unique_values = np.unique(y_train)
			mapper = dict(zip(unique_values, range(len(unique_values))))

			y_train = np.vectorize(lambda y: mapper[y])(y_train)

			model = create_model(x_train.shape[1] , len(mapper))	
			compile_model(model)

			model.summary()

			model.fit(
				x=x_train,
				y=y_train,
				epochs=10, 
				batch_size=4, 
				shuffle=True
			)

			model.save('models/agent.h5')