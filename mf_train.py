import pandas as pd
import os,time
import sys,csv
import matplotlib

import zipfile
import requests
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import dummy, metrics, cross_validation, ensemble
import keras.optimizers as optimizers
import keras.models as kmodels
import keras.layers as klayers
import keras.backend as K
import keras

start_time = int(time.time())
# Read in the dataset, and do a little preprocessing,
# mostly to set the column datatypes.
users = pd.read_csv('./data/users.csv', sep='::', 
                        engine='python', 
                        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']).set_index('UserID')
ratings = pd.read_csv('./data/train.csv', engine='python', 
                          sep=',', names=['TrainDataID', 'UserID', 'MovieID', 'Rating']).set_index('TrainDataID')
movies = pd.read_csv('./data/movies.csv', engine='python',
                         sep='::', names=['MovieID', 'Title', 'Genres']).set_index('MovieID')
testData = pd.read_csv('./data/test.csv', engine='python',
                         sep=',', names=['TestDataID', 'UserID', 'MovieID'])
users = users.drop(users.index[[0]])
ratings= ratings.drop(ratings.index[[0]])
movies = movies.drop(movies.index[[0]])
testData = testData.drop(testData.index[[0]])

# print(users)
# print(ratings)

movies['Genres'] = movies.Genres.str.split('|')

users.Age = users.Age.astype('category')
users.Gender = users.Gender.astype('category')
users.Occupation = users.Occupation.astype('category')
ratings.MovieID = ratings.MovieID.astype('category')
ratings.UserID = ratings.UserID.astype('category')
testData.MovieID = testData.MovieID.astype('category')
testData.UserID = testData.UserID.astype('category')

# Count the movies and users
n_movies = movies.shape[0]
n_users = users.shape[0]

mframes = [ratings.MovieID, testData.MovieID]
mref = pd.concat(mframes)

mres = pd.DataFrame(mref, columns = ['MovieID'] )
mres.MovieID = mres.MovieID.astype('category')

movieid = mres.MovieID.cat.codes.values
trainMovieID = movieid[0:ratings.MovieID.shape[0]]
testMovieID = movieid[ratings.MovieID.shape[0]:]
print(trainMovieID.shape)
print(testMovieID.shape)

uframes = [ratings.UserID, testData.UserID]
uref = pd.concat(uframes)

ures = pd.DataFrame(uref, columns = ['UserID'] )
ures.UserID = ures.UserID.astype('category')

userid = ures.UserID.cat.codes.values
trainUserID = userid[0:ratings.UserID.shape[0]]
testUserID = userid[ratings.UserID.shape[0]:]
print(trainUserID.shape)
print(testUserID.shape)


# y = np.zeros((ratings.shape[0], 5))
# y[np.arange(ratings.shape[0]), ratings.Rating - 1] = 1
y = np.array(ratings.Rating)

movie_input = keras.layers.Input(shape=[1])
movie_vec = keras.layers.Flatten()(keras.layers.Embedding(n_movies + 1, 128)(movie_input))
movie_vec = keras.layers.Dropout(0.5)(movie_vec)

# Users
user_input = keras.layers.Input(shape=[1])
user_vec = keras.layers.Flatten()(keras.layers.Embedding(n_users + 1,128)(user_input))
user_vec = keras.layers.Dropout(0.5)(user_vec)

# Bias vector
movie_bias = keras.layers.Embedding(n_movies + 1, 1, embeddings_initializer = 'random_normal')(movie_input)
movie_bias = keras.layers.Flatten()(movie_bias)
user_bias = keras.layers.Embedding(n_users +1 , 1, embeddings_initializer = 'random_normal')(user_input)
user_bias = keras.layers.Flatten()(user_bias)

# MF
r_hat = keras.layers.Dot(axes = 1)([ movie_vec, user_vec])
# r_hat = keras.layers.Add()(r_hat)

model = kmodels.Model([movie_input, user_input], r_hat)
opt = optimizers.adam(lr = 0.0003)
model.compile(opt, loss = 'mean_squared_error')
model.summary()

# Split  data
a_movieid, b_movieid, a_userid, b_userid, a_y, b_y = cross_validation.train_test_split(trainMovieID, trainUserID, y, test_size=0.05)

# earlystopping = EarlyStopping(monitor='val_loss', patience = 20, verbose=1, mode='min')
# checkpoint = ModelCheckpoint(filepath= str(start_time)+ 'modelfbest.h5',
#                             verbose=1,
#                             save_best_only=True,                            
#                             monitor='val_loss',
#                             mode='min') 
history = model.fit([a_movieid, a_userid], a_y, 
                        batch_size = 400,
                        epochs = 150,                         
                        validation_data=([b_movieid, b_userid], b_y))        
    # plot(history.history['loss'])
    # plot(history.history['val_loss'])

model.save(str(start_time) + 'mfmodel.h5')

scores = model.evaluate([a_movieid, a_userid], a_y, verbose=0)
print(scores)

scores = model.evaluate([b_movieid, b_userid], b_y, verbose=0)
print(scores)

result = model.predict([testMovieID, testUserID])
print(result)

with open(str(start_time) + 'result.csv' , "w", newline='') as mFile:
    writer = csv.writer(mFile)
    writer.writerow(["TestDataID","Rating"])
    for i in range(0, len(result)):
        mFile.write(str(i+1) + ",")        
        mFile.write(str(result[i][0]))
        mFile.write("\n")

