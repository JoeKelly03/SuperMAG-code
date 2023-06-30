#!/usr/bin/env python
# coding: utf-8

### script to perform an autoencoder,PCA Means shift and Agglomerative clustering on normalised Flux data


#import packages
import pdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy import stats
from mpl_toolkits.mplot3d import axes3d
import matplotlib
import matplotlib.colors as colors
from collections import Counter
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import estimate_bandwidth
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import math
from sklearn import mixture
from sklearn import cluster
from sklearn.decomposition import PCA
from matplotlib.colors import LogNorm
from matplotlib import ticker, cm
import pandas as pd
from datetime import datetime
from pathlib import Path
import os.path
import matplotlib.patches as mpatches
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift
from dateutil import parser
import glob
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from numpy.random import seed
from tensorflow.random import set_seed 
import cdflib
import cblind as cb
import time

from collections import defaultdict

from scipy.io import netcdf_file
import numpy as np
from matplotlib.ticker import FuncFormatter
import datetime as dt
from matplotlib.ticker import MultipleLocator
import os
import random
start_time = time.time()


##------------- Define Functions ----------------
##normalising
def normalise2(data):
  print('Normalising...')
  data[data==-1e31] = 0
  data[data==np.nan] = 0
  data[data==np.inf] = 0
  data[data==-np.inf] = 0
  copydata = data
  data_shape = data.shape
  data_norm_max = np.zeros(data_shape)
  #breakpoint()
  for i in range(0, len(data_norm_max)):
    data_max = data[i].max()
    PA = copydata[i]
    data_norm_max[i] = PA/data_max      #states a runtime warning message here from the division
  bad = np.isinf(data_norm_max)
  data_norm_max[bad==True] = 0
  bad = np.isnan(data_norm_max)
  data_norm_max[bad==True] = 0
  np.savez_compressed(os.path.join(motherpath,data_norm_savename), data_norm = data_norm_max)
  print("Maximum of the normalised data" , data_norm_max.max())
  return data_norm_max
  
#flattening data
def flatten_data(data_norm):
	'''function to flatten data if exists in 3D -> eg f(t,PA,E)
	input: normalised data in 3D
	output: normalised data in 2D '''
	print('Flattening...')
	data_flat = data_norm.reshape(len(data_norm),-1)
	np.savez_compressed(os.path.join(motherpath, data_flat_savename), data_flat = data_flat)
	return data_flat    

def split_data(data_flat, test_size, training_size):
	flat = data_flat
	print("Maximum of the flattened data" , data_flat.max())
	index = np.arange(0, len(data_flat),1)
	other, test, other_index, test_index = train_test_split(flat, index, test_size = test_size, random_state = 4)
	print("Length of flattened data" , len(data_flat))
	train, val, train_index, val_index = train_test_split(other, other_index, test_size = training_size, random_state = 4)
	print("Length of train set" , len(train) , "Length of test set" , len(test))
	np.savez_compressed(os.path.join(motherpath,test_savename), index = test_index, test = test)
	np.savez_compressed(os.path.join(motherpath,train_savename), index = train_index, train = train)
	np.savez_compressed(os.path.join(motherpath,val_savename), index = val_index, val = val)
	return train, val, test, test_index, val_index, train_index


##autoencoder
def autoencode_images(train, val, test, input_dims, encoded_dims):
	print('Autoencoding...') 
	print('training length:', len(train))
	print('validation length:', len(val))
	print('test length:', len(test))
#allows for the reproducability
	seed(1) 
#	set_seed(1) 
	

	#----- Build the Autoencoder-----------
	    
	#1) set up an input placeholder -> has the same dimensions as the flattened data. 
	#This defines the number of neurons in the input of the encoder.
	input_img = Input(shape=(input_dims,))
    
	#2) set up the encoded representation (reduction) (hidden layer)
	#This creates the hidden layer with the # of neurons = encoding dim and connects it to input
	encoded = Dense(encoded_dims, activation='relu')(input_img)
    
	#3) set up the decoded reconstruction (output layer)
	#This is reconstructing the input from the encoded layer. The number of neurons/ dimensions is the same as the input.
	decoded = Dense(input_dims, activation='sigmoid')(encoded)
    
	#4) create the model that maps the input to its reconstruction.
	# This builds the whole autoencoder: from the input image to the latent layer to the reconstruction.
	autoencoder = Model(input_img, decoded)
    
	#5) create the model that maps the input to its encoded representation (reduction)
	# This builds the network between the input and latent layer
	encoder = Model(input_img, encoded)
    
	#6) set up an encoder input placeholder (the hidden layer)
	#This defines the number of neurons in the encoded layer of the autoencoder. Acts as an 'input' in the reconstruction.
	encoded_input = Input(shape=(encoded_dims,))
    
	#7) retrieve the last layer of the autoencoder model
	#This defines the reconstruction layer of the autoencoder.
	decoder_layer = autoencoder.layers[-1]
    
	#8) create model that reconstructs the encoded image
	# This builds the network between the encoded layer and the reconstruction.
	decoder = Model(encoded_input, decoder_layer(encoded_input))#
  

	#9) Add any additional functions
	early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=5,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=False
    )
  
	#10) compile the autoencoder 
	#the metrics provide some statistical evaluation on the performance of the training on both the training set and validation set
	autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate = lrate), loss='mse')
 
	autoencoder.summary() # the number of params is the number of trainable weights
	#11) train the autoencoder
	#This will learn and identify patterns and relationships within the 'training' data set.
	#The epochs define the number of iterations, the batch_size is number of training samples to work with
	#Validation split sections of this amount of training data to test on the validate the results at the end of each epoch: 
	#The loss and model metrics are measured on this validation
	history = autoencoder.fit(train, train,
                epochs= epoch,
                batch_size= batch_size,
                shuffle=True,
                validation_data=(val,val), callbacks = [early_stopping]) 
                #last line helps model calculate val_loss and will stop when val_loss no longer increases/decreases											

	#12) apply the encoder to the test set AND then reconstuct
	#can apply to the full data set or the testing set
	#This is running the autoencoder
	encoded_imgs = encoder.predict(test) ## flat or test)
	decoded_imgs = decoder.predict(encoded_imgs) 
	

	#print(len(test), len(test_index),len(encoded_imgs))
    
	##13) save as a compressed file
	np.savez_compressed(os.path.join(motherpath, autoencoder_savename), encoded_imgs = encoded_imgs, decoded_imgs = decoded_imgs, training_loss = history.history['loss'], validation_loss = history.history['val_loss'])
    
	#14) plot the loss curves
 
	colour, linestyle = cb.Colorplots().cblind(2)
	plt.figure(figsize=(9,6))
	plt.plot(history.history['loss'],linewidth=2.5, c = colour[0])
	plt.plot(history.history['val_loss'],linewidth=2.5, c = colour[1])
	plt.ylabel('Loss value',fontsize=20)
	plt.xlabel('Epoch',fontsize=20)
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	plt.legend(['Training loss', 'Validation loss'], loc='upper right',fontsize=20)
	plt.grid()
	plt.title('SuperMAG loss curves'.upper())
    
	#save figure
	plt.savefig(os.path.join(motherpath, losscurve_figure_savename), format = 'png')
    
	plt.close()


	plt.figure(figsize=(9,6))
	plt.plot(history.history['loss'],linewidth=2.5, c = colour[0])
	plt.plot(history.history['val_loss'],linewidth=2.5, c = colour[1])
	plt.yscale('log')
	plt.ylabel('Loss value',fontsize=20)
	plt.xlabel('Epoch',fontsize=20)
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	plt.legend(['Training loss', 'Validation loss'], loc='upper right',fontsize=20)
	plt.grid()
	plt.title('SuperMAG log loss curves'.upper())
	#save figure
	plt.savefig(os.path.join(motherpath, log_losscurve_figure_savename), format = 'png')
    
	plt.close()


	print('Final loss:', history.history['val_loss'][-1])
	return encoded_imgs, decoded_imgs

#
##PCA
def pca_3d(encoded_imgs, PCA_dims):
	''' apply a 3D pca to the encoded images
	inputs:
	encoded_imgs: the encoded data from the AE
	PCA_dims: the number of PCA dimensions desired
    
	output:
	encoded_3d_pca: the applied 3D pca model to the encoded data'''
	#define evaluation method

	def evaluate_model(model, X):
		cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
		cross_scores = cross_val_score(model, X,)# scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
		return cross_scores
  
	print('PCA...')  
	#1) create the 3D pca model
	model_encoder = PCA(n_components=PCA_dims) #create the model of (p1,p2,p3)

	#2) apply to data and transform into PCA_dims 
	encoded_3d_pca = model_encoder.fit_transform(encoded_imgs) 

	#3) Evaluate
	params = model_encoder.get_params()
	#print(params)
	precision = model_encoder.get_precision()
	print('precision:',precision)
	score = model_encoder.score(encoded_imgs)
	print('score length', score.shape)
	print('score:',score)
	score_sample = model_encoder.score_samples(encoded_imgs)
	print('score sample length', score_sample.shape)
	print('score samples:',score_sample)

	var_ratio = model_encoder.explained_variance_ratio_
	print("Explained varience ratio", var_ratio[0:PCA_dims])
	#print(len(var_ratio))
	var = model_encoder.explained_variance_
	print("Explained varience" , var)
	noise = model_encoder.noise_variance_
	print("Noise" , noise)
	cross_scores = evaluate_model(model_encoder, encoded_imgs)
	cross_mean, cross_std = np.mean(cross_scores), np.std(cross_scores)
	print('log likelihood mean: %.3f , std: %.3f' %(cross_mean, cross_std))

	#4) plot
	fig = plt.figure(figsize=(15,15))
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(encoded_3d_pca[:,0],encoded_3d_pca[:,1],encoded_3d_pca[:,2], c = 'k')
	ax.set_xlabel('PCA_0',fontsize = 26, labelpad = 20)
	ax.set_ylabel('PCA_1',fontsize = 26, labelpad = 20)
	ax.set_zlabel('PCA_2',fontsize = 26, labelpad = 20)
	ax.xaxis.set_tick_params(labelsize=26, labelrotation = 45)
	ax.yaxis.set_tick_params(labelsize=26)
	ax.zaxis.set_tick_params(labelsize=26)
	ax.tick_params(axis='both', which='major', labelsize=26)
	ax.tick_params(axis='both', which='minor', labelsize=26)
	ax.view_init(7, 250)

	plt.title('SuperMAG PCA'.upper,fontsize = 20)
	plt.grid()    
    
	plt.savefig(os.path.join(motherpath ,PCA_figure_savename), format = 'png')
    
	plt.close()

	assert len(encoded_3d_pca)==len(encoded_imgs)
	dims = np.arange(0,encoded_dims,1) #I have a  feeling we are going to have to change the 102 here~I changed to 300

	fig, ax = plt.subplots(figsize=(10,10))
	#ax.bar(dims, var_ratio)
	#ax.scatter(dims, var_ratio)
	#ax.plot(dims, var_ratio, '--k')
	#ax.set_xlabel('# of dimensions')
	#ax.set_ylabel('Variance')
	#plt.savefig(os.path.join(motherpath ,Var_figure_savename), format = 'png')
	#plt.close()

	#fig, ax = plt.subplots(figsize=(10,10))
	#ax.scatter(dims, cross_scores)
	#ax.set_xlabel('# of dimensions')
	#ax.set_ylabel('CV Score')
	#plt.savefig(os.path.join(motherpath ,CV_figure_savename), format = 'png')
    
	#save as a compressed file	
	np.savez_compressed(os.path.join(motherpath, PCA_savename), encoded_3d_pca = encoded_3d_pca)#, score = score, score_sample = score_sample, params = params, precision = precision, variance = var, variance_ratio = var, noise = noise, cross_scores = cross_scores, cross_mean = cross_mean, cross_std = cross_std)
	return encoded_3d_pca

#bandwidth
def predict_bandwidth(encoded_3d_pca,n_samples,Q):
	''' function to estimate the badwidth for the Mean Shift algorithm'''
	print('Finding Bandwidth Estimate')
	st = time.process_time()
	bandwidth = estimate_bandwidth(encoded_3d_pca, quantile = Q, n_jobs = 2, random_state=0, n_samples = n_samples)
	print('bandwidth:', bandwidth)
	et = time.process_time()
	res = et - st
	print('CPU Execution time:', res, 'seconds')

	np.savez_compressed(os.path.join(motherpath, bandwidth_savename), bandwidth = bandwidth, Quantile = Q, N_Samples = n_samples)

	return bandwidth

#MeanShift
def Mean_Shift(encoded_3d_pca, bandwidth):
	'''function to perform the meanshift of the data.This will predict the number of clusters for agglomerative data.
	input:
	enocoded_3d_pca: the 3D flux data from the PCA

	output:
	ms_clustering: clustered data
	nclusters: number of clusters '''

	print('Mean Shift')
	st = time.process_time()
	# precict how many clusters
	ms_clustering = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=8).fit(encoded_3d_pca) 
	c = Counter(ms_clustering.labels_)
	clusters = sorted(c)
	nclusters = len(c)
	et = time.process_time()
	res = et - st
	print('CPU Execution time:', res, 'seconds')	

	print('number of clusters', nclusters)
	
	st = time.process_time()
	#sil_score = metrics.silhouette_score(encoded_3d_pca, ms_clustering.labels_, metric = 'euclidean', n_jobs=8) 
	#et = time.process_time()
	#res = et - st
	#print('Sil Score CPU Execution time:', res, 'seconds')	

	CH_score = metrics.calinski_harabasz_score(encoded_3d_pca, ms_clustering.labels_)
	DB_index = metrics.davies_bouldin_score(encoded_3d_pca, ms_clustering.labels_)
	#print('sil_score:', sil_score)
	print('CH_score:', CH_score)
	print('DB_index:', DB_index)


	#save
	np.savez_compressed(os.path.join(motherpath, MeanShift_savename), ms_clustering = ms_clustering.labels_, nclusters = nclusters, bandwidth = bandwidth, CH_score = CH_score, DB_index = DB_index)#, sil_score = sil_score)
	et = time.process_time()
	res = et - st
	print('CPU Execution time:', res, 'seconds')	

	ms_clustering = ms_clustering.labels_
	colour, linestyle = cb.Colorplots().rainbow(nclusters)
	patch =[]
	fig = plt.figure(figsize=(15,15))
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel('PCA 0',fontsize = 26, labelpad = 40)
	ax.set_ylabel('PCA 1',fontsize = 26, labelpad = 20)
	ax.set_zlabel('PCA 2',fontsize = 26, labelpad = 20)
	ax.xaxis.set_tick_params(labelsize=26, labelrotation = 45)
	ax.yaxis.set_tick_params(labelsize=26)
	ax.zaxis.set_tick_params(labelsize=26)
	ax.tick_params(axis='both', which='major', labelsize=26)
	ax.tick_params(axis='both', which='minor', labelsize=26)
	ax.view_init(7, 250)

	#3.2) sort data into clusters
	for n in range(0, nclusters):
		vars()['encoded_3d_pca_{}'.format(n)] = []

		for j in range(len(ms_clustering)):   
			if ms_clustering[j] == n:
                		vars()['encoded_3d_pca_{}'.format(n)].append(encoded_3d_pca[j])

		ax.scatter(np.array(vars()['encoded_3d_pca_{}'.format(n)])[:,0],np.array(vars()['encoded_3d_pca_{}'.format(n)])[:,1],np.array(vars()['encoded_3d_pca_{}'.format(n)])[:,2],label='Cluster {}'.format(n))     
		patch.append(mpatches.Patch(color=colour[n])) 

	#3.3) add extras and save
	plt.legend()#handles = patch)
	plt.title('Super MAG Mean Shift'.upper())
	plt.savefig(os.path.join(motherpath, ms_figure_savename), format = 'png')
	plt.close()

	return ms_clustering, nclusters

#Agglomerativeclustering
def Agglomerative(encoded_3d_pca, nclusters):
	'''function to perform the agglomerative of the data using the predicted the number of clusters from MeanShift.
	input:
	
	enocoded_3d_pca: the 3D flux data from the PCA
	nclusters: the number of clusters determined from Mean Shift

	output:
	agglomerative: the clustering data by agglomerative clustering'''

	print('Agglomerative')
	#1)cluster using agglomerative algorithm
	ac_clustering = AgglomerativeClustering(n_clusters=nclusters,linkage='ward').fit(encoded_3d_pca)

	ac_clustering = ac_clustering.labels_
	c = Counter(ac_clustering)
	print(c)

	#2) save
	np.savez_compressed(os.path.join(motherpath, agg_savename),ac_clustering)

	#3) Colour the PCA plot
	#3.1) define colour palette and build figure
	colour, linestyle = cb.Colorplots().rainbow(nclusters)
	patch =[]
	fig = plt.figure(figsize=(15,15))
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel('PCA 0')
	ax.set_ylabel('PCA 1')
	ax.set_zlabel('PCA 2')

	#3.2) sort data into clusters
	for n in range(0, nclusters):
		vars()['encoded_3d_pca_{}'.format(n)] = []

		for j in range(len(ac_clustering)):   
			if ac_clustering[j] == n:
                		vars()['encoded_3d_pca_{}'.format(n)].append(encoded_3d_pca[j])

		ax.scatter(np.array(vars()['encoded_3d_pca_{}'.format(n)])[:,0],np.array(vars()['encoded_3d_pca_{}'.format(n)])[:,1],np.array(vars()['encoded_3d_pca_{}'.format(n)])[:,2],label='Cluster {}'.format(n))     
		patch.append(mpatches.Patch(color=colour[n])) 

	#3.3) add extras and save
	plt.legend()#handles = patch)
	plt.title('SuperMAG PCA'.upper())
	plt.savefig(os.path.join(motherpath, agg_figure_savename), format = 'png')
	plt.close()

	return ac_clustering

def Kmeans_cluster(encoded_3d_pca,nclusters):
    ''' a function to apply a k means clustering algorithm to the 3D pca modelled data:
    enocoded_3d_pca: the 3D flux data from the PCA
nclusters: the number of clusters determined from Mean Shift'''

    print('kmeans')
    st = time.process_time()


    kmeans_model = KMeans(n_clusters= nclusters, random_state=0).fit(encoded_3d_pca) # cluster the data into n unknown groups depending on their p-unit vector quantities -> aka relative positions. 
    # clusters given by the estimation 

    kmeans = kmeans_model.labels_
    inertia = kmeans_model.inertia_
    centre_coords= kmeans_model.cluster_centers_
    #sil_score = metrics.silhouette_score(encoded_3d_pca, kmeans, metric = 'euclidean', n_jobs =8) 
    CH_score = metrics.calinski_harabasz_score(encoded_3d_pca, kmeans)
    DB_index = metrics.davies_bouldin_score(encoded_3d_pca, kmeans)
    print('inertia (coherence):', inertia)
    #print('sil_score:', sil_score)
    print('CH_score:', CH_score)
    print('DB_index:', DB_index)

    #save the classification
    np.savez_compressed(os.path.join(motherpath, kmeans_savename),kmeans = kmeans, inertia = inertia, cluster_centres = centre_coords, DB_index = DB_index, CH_score = CH_score)

    et = time.process_time()
    res = et - st
    print('CPU Execution time:', res, 'seconds')
    c = Counter(kmeans)
    clusters = sorted(c)
    print(c)
    #plot 
    #set the colour palette to be rainbow (cblind version)
    palette, linestyle = cb.Colorplots().rainbow(nclusters)
    patch =[]
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('PCA 0',fontsize = 26, labelpad = 40)
    ax.set_ylabel('PCA 1',fontsize = 26, labelpad = 20)
    ax.set_zlabel('PCA 2',fontsize = 26, labelpad = 20)
    ax.xaxis.set_tick_params(labelsize=26, labelrotation = 45)
    #ax.set_xticklables(rotation=90) ## the dates will appear sideways
    ax.yaxis.set_tick_params(labelsize=26)
    ax.zaxis.set_tick_params(labelsize=26)
    ax.tick_params(axis='both', which='major', labelsize=26)
    ax.tick_params(axis='both', which='minor', labelsize=26)
    ax.view_init(7, 250)
    for n in clusters:

        vars()['encoded_3d_pca_{}'.format(n)] = []

        for j in range(len(kmeans)):   
                if kmeans[j] == n:
                    vars()['encoded_3d_pca_{}'.format(n)].append(encoded_3d_pca[j]) #if in kmeans group =0, assign the encoded_img_3d value to the encoded_imgs_3d_0 array
               

        #x = np.array((vars()['encoded_3d_pca_{}'.format(n)])[:,0])#[0]
        #y = np.array((vars()['encoded_3d_pca_{}'.format(n)])[:,1])
        #z = np.array((vars()['encoded_3d_pca_{}'.format(n)])[:,2])

        ax.scatter(np.array(vars()['encoded_3d_pca_{}'.format(n)])[:,0],np.array(vars()['encoded_3d_pca_{}'.format(n)])[:,1],np.array(vars()['encoded_3d_pca_{}'.format(n)])[:,2], label='Cluster {}'.format(n))
        patch.append(mpatches.Patch(color=palette[n],label = '{}'.format(n))) 

    legend = ax.legend(handles= patch, loc = 'upper right', title = 'Cluster',fontsize= 26)
    plt.setp(legend.get_title(),fontsize= 26)


    plt.title('Super MAG K-means'.upper(),fontsize = 18)
    plt.tight_layout()
    ax.xaxis._axinfo['label']['space_factor'] = 5.0
    ax.yaxis._axinfo['label']['space_factor'] = 5.0
    ax.zaxis._axinfo['label']['space_factor'] = 5.0
    plt.savefig(os.path.join(motherpath, kmeans_figure_savename), format = 'png')
    plt.close()
    return kmeans


def time_stamp(kmeans,nclusters):

    #The purpose of this is to turn the list that contains a list of the cluster that a point is in (cluster_labels) and the epoch that contains the dates in alignment with the random order of the list that contains the clusters (aligned_epoch) into a dictionary that stores the dates of each cluster and using the cluster number as a key
    print("Creating the time stamps...")
    np.set_printoptions(suppress=True)
    cluster_labels = sorted(kmeans)
    #Just checking the first 5 datapoints clusters
    #print("Kmeans sample",kmeans[:10])
    #Now concatonating all the time variables into one variable
    datetime_list = []

    # Iterate over the time variables
    for month, day, hour, minute, second in zip(datamo.astype(int), datady.astype(int), datahr.astype(int), datamt.astype(int), datasc.astype(int)):
        dt = datetime(year=2010, month=month, day=day, hour=hour, minute=minute,  second=second)
        datetime_list.append(dt)   #This is now the epoch_all variable
    epoch = sorted(datetime_list)
    #print("Sample of the epoch", epoch[:10])

    #Now using the same random seed as in the ML code to get the correct random dates that align with the data with the clusters however it also randomises the order I am not sure if the original data does this aswell
    index = np.arange(0, len(datetime_list),1)
    other, test, other_index, test_index = train_test_split(epoch, index, test_size = test_size, random_state = 4)
    aligned_epoch = test
    #print("Sampled of the aligned epoch", aligned_epoch[:10])

    #Now to get the clusters
    clusters = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        data_point = aligned_epoch[i]
        clusters[label].append(data_point)
        
    # Create a dictionary to store the dates of each cluster
    cluster_dates = {}
    for cluster_number, data_points in clusters.items():
        dates_for_cluster = [data_point for data_point in data_points]
        cluster_dates[cluster_number] = dates_for_cluster
        
        
        
#    # Create a dictionary to store the labeled data points
#    labeled_data = {}
#    for i, d in enumerate(test):
#        key = d.strftime('%Y-%m-%d %H:%M:%S')
#        labeled_data[key] = i
#
#    # Create a dictionary to store the clusters
#    clusters = defaultdict(list)
#    for i, label in enumerate(cluster_labels):
#        data_point = test[i]
#        clusters[label].append((data_point, i))  # Store the data point along with its index
#
#    clusters_indices = {}
#    dates = {}
#    clustersize = {}
#    # Access labeled data points in each cluster
#    for cluster_number, data_points in clusters.items():
#        print("Cluster {}: ".format(cluster_number))
#        clusters_indices[cluster_number] = [index for _, index in data_points]  # Extract the indices from the data points
#        #print("Data Points: ", [data_point for data_point, _ in data_points])
#        #print("Indices: ", clusters_indices[cluster_number])
#        cluster_array = np.array(clusters_indices[cluster_number])
#        cluster_shape = cluster_array.shape
#        clustersize[cluster_number] = cluster_shape
#        print(cluster_shape)
#        #Code for the dates of all the datapoints in each cluster
#        datesforcluster = []
#        for time in range(0,cluster_shape[0]):
#            date = epoch[clusters_indices[cluster_number][time]]
#            datesforcluster.append(date)
#        dates[cluster_number] = datesforcluster
    
    # Create an empty dictionary to store the sizes of each cluster
    clustersize = {}

# Iterate through the 'cluster_dates' dictionary and calculate the size of each cluster
    for cluster_number, dates_list in cluster_dates.items():
        clustersize[cluster_number] = len(dates_list)

# Now 'cluster_sizes' contains the size (number of data points) for each cluster
    print("Cluster Sizes:", clustersize)
    
    np.savez_compressed(os.path.join(motherpath, dates_savename),cluster_dates)
    return cluster_dates, clustersize
    
def geo_plots(cluster_dates, clustersize):
    print("Creating geographical plots...")
    def format_mlt():
        """Return MLT in hours rather than a number of degrees when drawing axis labels."""
 
        def formatter_function(y, pos):
            hours = y * (12 / np.pi)
            if hours == 24:
                return ""
            else:
                if hours < 0:
                    hours += 24
                return "{:.0f}".format(hours)
 
        return FuncFormatter(formatter_function)
        
    def configure_polar_plot(ax, rmax, colat_grid_spacing=10, theta_range=None, mlt=True):
        """Configures a polar plot to appear on the page correctly."""
        # Configure colatitude.
        ax.set_rmin(0.0)
        ax.set_rmax(rmax)
        ax.yaxis.set_major_locator(MultipleLocator(colat_grid_spacing))
        ax.set_theta_zero_location("S")
        if theta_range is not None:
            ax.set_thetamin(theta_range[0])
            ax.set_thetamax(theta_range[1])
        if mlt:
            ax.xaxis.set_major_formatter(format_mlt())
            ax.xaxis.set_major_locator(MultipleLocator(np.pi / 2))
        ax.grid(True)
        
    def read(datetime):
        # Load the data from the netCDF file
        with netcdf_file(datapath / f"{datetime:%Y%m%d}.north.schavec-mlt-supermag.60s.rev-0006.ncdf") as f:
            lat = f.variables["mlat"][:].copy()
            mlt = f.variables["mlt"][:].copy()
            dbn_nez = f.variables["dbn_nez"][:].copy()
            years = f.variables["time_yr"][:].astype(int)
            months = f.variables["time_mo"][:].astype(int)
            days = f.variables["time_dy"][:].astype(int)
            hours = f.variables["time_hr"][:].astype(int)
            minutes = f.variables["time_mt"][:].astype(int)
            seconds = f.variables["time_sc"][:].astype(int)
        time = []
        for cnt, year in enumerate(years):
            time.append(dt.datetime(year, months[cnt], days[cnt], hours[cnt], minutes[cnt], seconds[cnt]))
        time = np.array(time)
        
        return time, lat, mlt, dbn_nez
        
    def make_map_for_timestamp(datetime, time, lat, mlt, dbn_nez):    
        time_index = np.where(time == datetime)[0][0]
        colat = 90 - lat[time_index, :].reshape(24, 25)[0, :]                 # matplotlib expects colatitude, so convert
        theta = mlt[time_index, :].reshape(24, 25)[:, 0] * np.pi / 12         # angle = mlt * pi / 12 (i.e. from 24 MLT to 2pi radians)
        z = dbn_nez[time_index, :].reshape(24, 25).T                          # use .T to stop matplotlib complaining
        clim = np.max(np.abs(z))                                              # make sure the colour bar is symmetrical about 0
        fig, ax = plt.subplots(subplot_kw={"polar": True})
        mesh = ax.pcolormesh(theta, colat, z, shading="nearest", vmin=-clim, vmax=clim)
        cbar = plt.colorbar(mesh)
        cbar.set_label("dB (nT)")
        configure_polar_plot(ax, 50)
        return fig, ax
    


    c = 0
    for cc in range(0,nclusters):
      folder_name = f"cc_{cc}"  # Create a folder name based on cc value   
      # Create the folder if it doesn't exist
      folder_path = os.path.join("./geoplots", folder_name)
      os.makedirs(folder_path, exist_ok=True)
      previous_time = None
      
      #alldates = []
      
      for timestamps in range(0,12):  #clustersize[cc]#if you wanted all of the plots  #the index of the time you want within the cluster Changetimehere
        c = c+1
        timestamp = cluster_dates[cc][timestamps]
        year = timestamp.year
        month = timestamp.month
        day = timestamp.day
        hour = timestamp.hour
        minutes = timestamp.minute
        datetime_obj = dt.datetime(year, month, day, hour, minutes)

        
        #alldates.append(datetime_obj)    #This part was just a small attempt at getting the average plots for clusters
        #av_figure, av_ax = make_map_for_timestamp(datetime_obj, time, lat, mlt, b_data)
        
        #if month != previous_time:   #This part ended up not working as the shuffle from the randomised train set test set reshuffles all the data   
        time, lat, mlt, b_data = read(datetime_obj)
        figure, ax = make_map_for_timestamp(datetime_obj, time, lat, mlt, b_data)
        filename = datetime_obj.strftime("%Y%m%d%H%M.png")
        title = f"Cluster: {cc} -Timestamp: {datetime_obj}"
        plt.title(title.upper())       
        plt.savefig(os.path.join(folder_path, filename), format = 'png')
        plt.close(figure)
            
        #previous_time = month 
              

#          fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
#          for i, plot in enumerate(plots):
#            row = i//3
#            col = i%3
#            axs[row, col].imshow(plot)
#          fig.tight_layout()
#          plt.savefig("combined_plots.png", format='png')      #Snippet of code for potentially plotting lots of plots in the same png for easier viewing
        
        
    return c
#

## ----------- Adjustable Params -----------------

#split data
random_state = 4 # set the random seed
test_size = 0.6 # want 60% of the original data to be our test set
training_size = 0.5 #want 50% of the remaining 40% to be our training set and 50% to be validation set - 20:20:60 split

#autoencoder
input_dims = 600  #number of dimensions of the input dataset 600
encoded_dims = 300 #number of dimensions of the hidden, encoded layer Shannon picked half
lrate = 0.0005 # learning rate of the optimizer. This will need adjusting
epoch = 100    # number of iterations the autoencoder will train for Shannon picked 500 but that takes a long time. This will need adjusting
batch_size = 512   # number of samples that are propagated through the network each iteration. Determined by 2^n, where n is positive. Should be as close to the # of dimensions as possible.
#validation = 1/8 # there fraction of points selected to be excluded from the training and are therefore tested on at the end of each epoch. To be used instead of a validation dataset

#PCA
PCA_dims = 3 # set to 3 for visualisation purposes

#MeanShift
# to estimate bandwidth
n_samples = 1000  #the size of the dataset used to predict the bandwidth so larger is going to be more accurate should be the same size as our dataset but it has to be balanced with computational cost
#print('all samples')
Q = 0.25  #quantile or the range that the bandwidth estimation will consider data within that percentage. This will need adjusting
print('Quantile:',Q)
print('bandwidth samples:',n_samples)

#We will have to manullay adjust and observe the lrate, epoch and q for our data

#nclusters = 5    #usually commented out as we dont want to force a number of clusters which would make it supervised machine learning

#to manually define bandwidth
#bandwidth = 6.00951
#print('bandwidth = {}'.format(bandwidth))

##	#------------ Directories -----------------------
	#data
motherpath = './motherpathSU'
data_path = './2010data/data_all.npz'
datapath = Path('./2010')
#
  
#	#-------------- File Savenames ---------------	
data_norm_savename = 'norm_max_data'
data_flat_savename = 'data_flat'
test_savename = 'test_data'
train_savename = 'train_data'
val_savename = 'val_data'

autoencoder_savename = 'autoencoder_outputs'
PCA_savename = '3D_PCA'
bandwidth_savename = 'bandwidth'
MeanShift_savename = 'MeanShift'
kmeans_savename ='kmeans'

losscurve_figure_savename =  'Loss_curves.png'
log_losscurve_figure_savename =  'Log_Loss_curves.png'
PCA_figure_savename = '3D_PCA.png'
VAR_figure_savename = '3D_PCA_VAR.png'
CV_figure_savename = '3D_PCA__CV.png'
ms_figure_savename = 'Meanshift.png'
kmeans_figure_savename = 'kmeans.png'
dates_savename = 'dates.npz'

#	#-------------- DATA ---------------------------
#Here you can load in data that you know is correct for speed (might be pickleing)
#data_all_file = np.load(data_path,allow_pickle=True)['dbn_nez_all']
#Loading in the times and ID for the data for later
datamo = np.load(data_path, allow_pickle=True)['time_mo_all']
datady = np.load(data_path, allow_pickle=True)['time_dy_all']
datahr = np.load(data_path, allow_pickle=True)['time_hr_all']
datamt = np.load(data_path, allow_pickle=True)['time_mt_all']
datasc = np.load(data_path, allow_pickle=True)['time_sc_all']

#This section is for when you want to use multiple variables.-------
#keys = data_all_file.files
#arrays = []
#for key in keys:
#  array = data_all_file[key]
#  arrays.append(array)
#  
#for key, array in zip(keys, arrays):
#    print(f"Array {key} dimensions: {array.shape}")
#  
#data_all = np.concatenate(arrays)
#------------------------------------------

data_all=np.load(data_path,allow_pickle=True)['dbn_nez_all'] #This was the code given that the data is a dictionary
#data_norm_max = np.load(os.path.join(motherpath, 'norm_max_data.npz'),allow_pickle=True)['data_norm_max']
#data_flat = np.load(os.path.join(motherpath, 'norm_max_data_flat.npz'),allow_pickle=True)['data_flat']
#train = np.load(os.path.join(motherpath, 'train_data.npz'),allow_pickle=True)['train']
#val = np.load(os.path.join(motherpath, 'val_data.npz'),allow_pickle=True)['val']
#test = np.load(os.path.join(motherpath, 'test_data.npz'),allow_pickle=True)['test']
#encoded_imgs = np.load(os.path.join(motherpath, 'autoencoder_outputs.npz'), allow_pickle = True)['encoded_imgs']
#encoded_3d_pca = np.load(os.path.join(motherpath, '3D_PCA.npz'), allow_pickle=True)['encoded_3d_pca']
#bandwidth = (np.load(os.path.join(motherpath, 'bandwidth.npz'), allow_pickle=True)['bandwidth']).item() 
#nclusters = np.load(os.path.join(motherpath, 'MeanShift.npz'), allow_pickle=True)['nclusters']
#ms_clustering = np.load(os.path.join(motherpath, 'MeanShift.npz'), allow_pickle=True)["ms_clustering"]


#
#	#-------------- Data Slicing ----------------
#	# cut to Ebin = 2.6MeV
## ---------- RUN --------------
#	Stage 1
data_norm = normalise2(data_all)
data_flat = flatten_data(data_norm)
print('Splitting data...')
train, val, test, train_index, val_index, test_index = split_data(data_flat, test_size, training_size)
encoded_imgs, decoded_imgs = autoencode_images(train,val, test, input_dims, encoded_dims)
encoded_3d_pca = pca_3d(encoded_imgs, PCA_dims)
bandwidth = predict_bandwidth(encoded_3d_pca, n_samples, Q)

#Stage 2
ms_clustering, nclusters = Mean_Shift(encoded_3d_pca, bandwidth)
print(n_samples, Q)
####ag_clustering = Agglomerative(encoded_3d_pca, nclusters) #This is too expensive for the memory
kmeans = Kmeans_cluster(encoded_3d_pca, nclusters)
cluster_dates, clustersize= time_stamp(kmeans, nclusters)  #You can use either clustering algorithm but kmeans may be better.
geographicplots = geo_plots(cluster_dates, clustersize)

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time, "seconds")
		