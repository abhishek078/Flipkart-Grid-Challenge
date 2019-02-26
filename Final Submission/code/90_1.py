import pandas as pd
import numpy as np
import keras.backend as K
from keras import metrics
import argparse
import scipy
np.random.seed(5)
from skimage import color
from skimage import io
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from PIL import Image
import os


from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
import cv2

def image_handler(path):
	filename = '../data/images/' + path
	im = cv2.imread(filename)
	im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	im = cv2.resize(im, (0,0), fx=0.25, fy=0.25) 
	return im


def build_model():
	input = Input(shape=(120, 160, 1))

	# Layer 1
	x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input)
	x = BatchNormalization(name='norm_1')(x)
	x = LeakyReLU(alpha=0.1)(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)

	# Layer 2

	x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
	x = BatchNormalization(name='norm_2')(x)
	x = LeakyReLU(alpha=0.1)(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)

	# Layer 3
	x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
	x = BatchNormalization(name='norm_3')(x)
	x = LeakyReLU(alpha=0.1)(x)

	# Layer 4
	x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
	x = BatchNormalization(name='norm_4')(x)
	x = LeakyReLU(alpha=0.1)(x)

	# Layer 5
	x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
	x = BatchNormalization(name='norm_5')(x)
	x = LeakyReLU(alpha=0.1)(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)

	# Layer 6
	x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
	x = BatchNormalization(name='norm_6')(x)
	x = LeakyReLU(alpha=0.1)(x)

	# Layer 7
	x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
	x = BatchNormalization(name='norm_7')(x)
	x = LeakyReLU(alpha=0.1)(x)

	# Layer 8
	x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
	x = BatchNormalization(name='norm_8')(x)
	x = LeakyReLU(alpha=0.1)(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)

	# Layer 9
	x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
	x = BatchNormalization(name='norm_9')(x)
	x = LeakyReLU(alpha=0.1)(x)

	# Layer 10
	x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
	x = BatchNormalization(name='norm_10')(x)
	x = LeakyReLU(alpha=0.1)(x)

	flatten = Flatten() (x)
	dense1 = Dense(256, activation = 'tanh') (flatten)
	
	output1 = Dense(1, activation = 'sigmoid') (dense1) 
	output2 = Dense(1, activation = 'sigmoid') (dense1)
	output3 = Dense(1, activation = 'sigmoid') (dense1)
	output4 = Dense(1, activation = 'sigmoid') (dense1) 

	model = Model(inputs= input , outputs=[output1, output2, output3, output4])
	model.summary()
	return model


def iou(bxs,bws,bys,bhs,y_pred_bx,y_pred_bw,y_pred_by,y_pred_bh):
	box1=[]
	box2=[]
	box1.append(bxs)
	box1.append(bys)
	box1.append(bxs+bws)
	box1.append(bys+bhs)
	box2.append(y_pred_bx)
	box2.append(y_pred_by)
	box2.append(y_pred_bx+y_pred_bw)
	box2.append(y_pred_by+y_pred_bh)

	xi1 = max(box1[0],box2[0])
	yi1 = max(box1[1],box2[1])
	xi2 = min(box1[2],box2[2])
	yi2 = min(box1[3],box2[3])
	inter_area = (yi2-yi1)*(xi2-xi1)
    
	box1_area = (box1[3]-box1[1])*(box1[2]-box1[0])
	box2_area = (box2[3]-box2[1])*(box2[2]-box2[0])
	union_area = box1_area+box2_area-inter_area
    
	iou = inter_area/union_area
	return iou



def train_build(filename):
	# load the dataset
	TRAIN_SET_PATH = filename
	data = open(TRAIN_SET_PATH).readlines()

	X, Y = [] , []
	bxs, bys, bws, bhs = [], [] , [], [],

	for line in data:
		imagePath, bx, bw, by, bh = line.strip().split(",")
		#X.append(np.array(image_process(imagePath.strip())).flatten())
		bx = float(bx) / 640.0
		bw = float(bw) / 640.0
		by = float(by) / 480.0
		bh = float(bh) / 480.0
		bxs.append(bx)
		bws.append(bw)
		bys.append(by)
		bhs.append(bh)
		#Y.append( [ bx, bw, by, bh ] )
	
	#X = np.array(X)
	#np.save('X_data_train', X)
	X = np.load('X_pest.npy')
	X = X.reshape(-1, 120, 160, 1)

	#Y = np.array(Y)
	bxs = np.array(bxs)
	bws = np.array(bws)
	bys = np.array(bys)
	bhs = np.array(bhs)
	
	return (X, bxs, bws, bys, bhs)



def train():


	model = build_model()
	#model.load_weights('90_1.h5')
	model.compile(optimizer = 'adadelta' , loss = ['mean_squared_error', 'mean_squared_error', 'mean_squared_error', 'mean_squared_error'] , metrics=[])
	
	
	X, bxs, bws, bys, bhs = train_build('../data/train_out.csv')
	
	#X_train, Y_train_bx, Y_train_bw, Y_train_by, Y_train_bh = train_build('../data/train.csv')

	#X_val, Y_val_bx, Y_val_bw, Y_val_by, Y_val_bh = val_build('../data/val.csv')
	
	X_train, X_val, Y_train_bx, Y_val_bx , Y_train_bw, Y_val_bw , Y_train_by, Y_val_by , Y_train_bh, Y_val_bh = train_test_split(X, bxs, bws, bys, bhs, test_size = 0.1, random_state=5)
	
	optimizer = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-08, decay=0.0)

	
	checkpoint_callback = ModelCheckpoint('90_1.h5', verbose=1, save_weights_only=True, save_best_only=True, mode='min')
	epochs = 200
	batch_size = 32
	history = model.fit(X_train, [Y_train_bx, Y_train_bw, Y_train_by, Y_train_bh ], batch_size = batch_size, epochs = epochs, validation_data = (X_val, [Y_val_bx, Y_val_bw, Y_val_by, Y_val_bh]), verbose = 2, callbacks=[checkpoint_callback])



def test():

	model_test = build_model()
	model_test.load_weights('90_1.h5')


	'''TEST_SET_PATH = "../data/test.csv"
	data = open(TEST_SET_PATH).readlines()[1:]

	X_test = []

	for line in data:
		imagePath, bx, bw, by, bh = line.strip().split(",")
		#X_test.append(np.array(image_process(imagePath.strip())).flatten())
	

	#X_test = np.array(X_test)
	#np.save('X_test', X_test)
	X_test = np.load('X_test.npy')
	X_test = X_test.reshape(-1, 120, 160, 1)
	y_pred_bx , y_pred_bw, y_pred_by, y_pred_bh  = model_test.predict(X_test)
	
	y_pred_bx *= 640.0
	y_pred_bw *= 640.0
	y_pred_by *= 480.0
	y_pred_bh *= 480.0

	y_pred_bx = y_pred_bx.astype('int32')
	y_pred_bw = y_pred_bw.astype('int32')
	y_pred_by = y_pred_by.astype('int32')
	y_pred_bh = y_pred_bh.astype('int32')

	y_pred_bx = y_pred_bx.flatten()
	y_pred_bw = y_pred_bw.flatten()
	y_pred_by = y_pred_by.flatten()
	y_pred_bh = y_pred_bh.flatten()

	#Generate Submisions Ouptput Csv File
	PATH = "../data/"
	predictions = {'x1': y_pred_bx, 'x2': y_pred_bw, 'y1': y_pred_by, 'y2': y_pred_bh }
	sub_df = pd.DataFrame(data = predictions)

	test_submission = pd.read_csv(TEST_SET_PATH)
	submission_df = test_submission[['image_name']]

	submission_df = submission_df.join(sub_df)
	submission_df.to_csv(PATH + "sub_simple_3:17.csv", index=False)
	print("****************************")
	print("submission file is created with name sub_simple_s3.csv in ../data/ folder")'''

	TRAIN_SET_PATH = "../data/train.csv"
	data = open(TRAIN_SET_PATH).readlines()[1:]
	fwrite = open('train_out_90.csv', 'w')
	bxs=[]
	bws=[]
	bys=[]
	bhs=[]
	
	for line in data:
			imagePath, bx, bw, by, bh = line.strip().split(",")
			bx=float(bx)
			bx=int(bx)
			bxs.append(bx)
			bw=float(bw)
			bw=int(bw)
			bws.append(bw)
			by=float(by)
			by=int(by)
			bys.append(by)
			bh=float(bh)
			bh=int(bh)
			bhs.append(bh)

	bxs = np.array(bxs, dtype=np.int32)
	bws = np.array(bws, dtype=np.int32)
	bys = np.array(bys, dtype=np.int32)
	bhs = np.array(bhs, dtype=np.int32)

	bxs = bxs.flatten()
	bws = bws.flatten()
	bys = bys.flatten()
	bhs = bhs.flatten()

	X_test = np.load('X_data_train.npy')
	X_test_data = np.copy(X_test)
	X_test = X_test.reshape(-1, 120, 160, 1)
	y_pred_bx , y_pred_bw, y_pred_by, y_pred_bh  = model_test.predict(X_test)

	y_pred_bx *= 640.0
	y_pred_bw *= 640.0
	y_pred_by *= 480.0
	y_pred_bh *= 480.0

	y_pred_bx = y_pred_bx.astype('int32')
	y_pred_bw = y_pred_bw.astype('int32')
	y_pred_by = y_pred_by.astype('int32')
	y_pred_bh = y_pred_bh.astype('int32')

	y_pred_bx = y_pred_bx.flatten()
	y_pred_bw = y_pred_bw.flatten()
	y_pred_by = y_pred_by.flatten()
	y_pred_bh = y_pred_bh.flatten()


	count = 0
	#X_pest=[]
	for i in range(bxs.size):
		v=iou(bxs[i],bws[i],bys[i],bhs[i],y_pred_bx[i],y_pred_bw[i],y_pred_by[i],y_pred_bh[i])
		if v>0.75:
			#print(v)
			fwrite.write(data[i])
			count += 1
			#s=X_test_data[i]
			#s=s.flatten()
			X_pest.append(X_test_data[i])

	X_pest=np.array(X_pest)
	np.save('X_pest_90', X_pest)
	print(count)

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", type=str)
	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = get_args()
	if args.mode == "train":
		train()
	elif args.mode == "test":
		test()
	else:
		print("wrong mode")
