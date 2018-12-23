# Movie-Recommender-System

## Goal
Based on user-movie-rating triple, predict the rating of unseen user-movie pair. Explore different methods, such as Matrix Factorizaton and NN-based methods, and complare the results.

## Data

### User
|UserID|Gender|Age|Occupation|Zip-code|
|:-:|:-:|:-:|:-:|:-:|
|796|F|1|10|48067|
|3203|M|56|16|70072|
|4387|M|25|15|55117|


### Movie
movieID|Title|Genres
|:-:|:-:|:-:|
|1|Toy Story (1995)|Animation|Children's|Comedy|
|2|Jumanji (1995)|Adventure|Children's|Fantasy|
|3|Grumpier Old Men (1995)|Comedy|Romance|
### User-Movie-Rating Triple
|TrainDataID|UserID|MovieID|Rating|
|:-:|:-:|:-:|:-:|
|1|796|1193|5|
|2|796|661|3|
|3|796|914|3|
|4|796|3408|4|

## Results
### Baseline on [Kaggle](https://www.kaggle.com/c/ml2017-hw6/leaderboard)
* Strong baseline: RMSE 0.87389
* Simple baseline: RMSE 0.93104

### Matrix Factorization
Model Settings

```
dimension 32, learning rate 0.0003, 175 epoch → RMSE = 0.73801263
dimension 64, learning rate 0.0003, 175 epoch → RMSE = 0.71441079
dimension 84, learning rate 0.0003, 175 epoch → RMSE = 0.7196014
dimension 128 , learning rate 0.0003, 175 epoch → RMSE = 0.715795
```
Best result: RMSE = 0.71441079

### DNN
Best result: RMSE = 0.86614

### Comparison
Both methods are better than the strong baseline. However, after experimenting different model settings, MF methods almost always beats DNN. Maybe should try RNN next time.

## Visulization
<img src="https://i.imgur.com/R0NR95T.png">

T-sne components of movie embeddings 

```
Red :["Children's", "Musical", "Animation" , 'Documentary','Comedy']
Green :['War', 'Crime', 'Sci-Fi','Action', 'Adventure']
Blue :[ 'Drama', 'Romance']
Purple:[ 'Fantasy','Thriller', 'Horror' ]
```
