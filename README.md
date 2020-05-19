# Alternating Least Squares

This ALS.py class is an implementation of "Collaborative filtering for implicit feedback datasets" by Koren, which is factorizing implicit dataset by alternating least squares.



For test, I used MovieLens 25M data, which consists of ratings on 62,000 movies by 162,000 users

(https://grouplens.org/datasets/movielens/25m/). For simplicity, I used only part of data.

```python
# read and sample part of data
dataset = 'ml-25m/ratings.csv'
ratings = pd.read_csv(dataset).sample(n = 30000, random_state= 1)

# add binary for implicit
ratings['binary'] = 1

ratings.head()
```



<p align ='center'>
    <img src = "../img/data_head.png" style="width: 60%"> <br/>
    <sub>Sample Ratings</sub>
</p>


From data, extract users who rated at least four movies.



```python
user_rating_counts = ratings.groupby('userId').rating.count()
user_rating_counts = user_rating_counts[user_rating_counts>=4]
ratings = ratings.merge(user_rating_counts, how = 'right', left_on='userId', right_index=True)
```



Transform data into a matrix form.


```python
# get categories of unique users and unique movies
user_c = CategoricalDtype(sorted(ratings.userId.unique()), ordered=True)
movie_c = CategoricalDtype(sorted(ratings.movieId.unique()), ordered=True)

# transform each ratings to categories of users and movies
row = ratings.userId.astype(user_c).cat.codes
col = ratings.movieId.astype(movie_c).cat.codes

# create binary rating matrix out of row and col data
R = csr_matrix((ratings["binary"], (row, col)), \
                           shape=(user_c.categories.size, movie_c.categories.size))

R
```

```
<441x1857 sparse matrix of type '<class 'numpy.int64'>'
	with 2192 stored elements in Compressed Sparse Row format>
```

There are 441 users with 1857 movies.



```python
# create ALS instance
als = ALS()

# fit train data
als.fit(R, hidden_size = 100, iteration = 3)
```

```
cost: 90817901.60814771
iter: 0
User Matrix
[==================================================] 100%
Movie Matrix
[==================================================] 100%
cost: 6407.207414920185
iter: 1
User Matrix
[==================================================] 100%
Movie Matrix
[==================================================] 100%
cost: 6095.757967740575
iter: 2
User Matrix
[==================================================] 100%
Movie Matrix
[==================================================] 100%
cost: 5933.454209038446
iter: 3
User Matrix
[==================================================] 100%
Movie Matrix
[==================================================] 100%
cost: 5830.76029880032
5830.76029880032
```



For prediction, I utilized `movies.csv`, which consists of movieId, title, and genre.

```python
# read and make dictionary
movies = pd.read_csv('ml-25m/movies.csv')
movie_title_dict = {movies.movieId[i]:[movies.title[i], movies.genres[i]] for i in range(movies.shape[0])}

# movies person i watched
i = 200
recommendations = als.predict(i)

print('Watched')
for watched_movie in R[i].indices:
    movie_idx = movie_c.categories[watched_movie]
    print(movie_title_dict[movie_idx])
    
print('\nRecommended')
for each_movie in recommendations:
    if each_movie not in R[i].indices:
        movie_idx = movie_c.categories[each_movie]
        print(movie_title_dict[movie_idx])
```



```
Watched
['Apollo 13 (1995)', 'Adventure|Drama|IMAX']
['Jurassic Park (1993)', 'Action|Adventure|Sci-Fi|Thriller']
['Dear God (1996)', 'Comedy']
['Japanese Story (2003)', 'Drama']
['De-Lovely (2004)', 'Drama|Musical']

Recommended
['Die Hard: With a Vengeance (1995)', 'Action|Crime|Thriller']
['Tootsie (1982)', 'Comedy|Romance']
['Wizard of Oz, The (1939)', 'Adventure|Children|Fantasy|Musical']
["William Shakespeare's Romeo + Juliet (1996)", 'Drama|Romance']
['Annie Hall (1977)', 'Comedy|Romance']
```

