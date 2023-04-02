library("dslabs")
library("ggplot2")
library("dplyr")
library("tidyverse")
library("gtools")
library("lubridate")
library("Lahman")
library("broom")
library("caret")
library("gam")
library("rpart")
library("tinytex")
library("recommenderlab")
library("reshape2")
library("recosystem")
library("data.table")

##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)

#------------------------------------------------------------------------------

##Wrangling the data____________________________________________________

## We start by inspection the values and column classes
first_inspection <- print(head(edx, 10))
dataset_Classes <- sapply(edx,class)
print(dataset_Classes)
#Inspect the number of unique values for title and genres

unique_movieIds <- n_distinct(edx$movieId)
unique_userIds <- n_distinct(edx$userId)
unique_genres <- n_distinct(edx$genres)
tbl <- data.frame(
  'Variable' = c('movieId', 'userId', 'genres'),
  'Unique Values' = c(unique_movieIds, unique_userIds, unique_genres)
)

print(tbl)

#inspect for NA values
print(sum(is.na(edx)))

#separate the title column into 2 different variables. 
#Then convert the year characters into integer values.

edx <- edx %>%
  mutate(year = as.integer(str_extract(title, "(?<=\\()\\d{4}(?=\\))")))

edx <- edx %>%
  mutate(title = str_replace(title, " \\(\\d{4}\\)", ""))

#extract all the unique values of the "genre" vector in 1 column

mini_edx <- edx[1:10000,1:6]

mini_edx2 <- mini_edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres)
genres <- unique(mini_edx2$genres)


# Create an empty data frame with columns for each genre
edx2 <- data.frame(matrix(0, ncol = length(genres), nrow = nrow(edx)))
names(edx2) <- genres

# Loop through each genre and add a 1 to the corresponding column if the genre is present
for (genre in genres) {
  edx2[, genre] <- as.integer(grepl(genre, edx$genres))
}

#add the genre binary columns to edx data-set

edx2 <- cbind(edx, edx2)

#remove the "genres" column from the "edx2" (twin data-set) data frame.

edx2 <- edx2[, !(names(edx2) %in% c("genres"))]


#Conver "timestamp into a more understandable variable. In this case, dates

edx <- edx %>% mutate(date = as_datetime(timestamp))
edx2 <- edx2 %>% mutate(date = as_datetime(timestamp))

##We add a new column to both edx and edx2 data frames (number of ratings per movie).

edx <- edx %>%
  group_by(movieId) %>%
  mutate(num_ratings = n()) %>%
  ungroup()

edx2 <- edx2 %>%
  group_by(movieId) %>%
  mutate(num_ratings = n()) %>%
  ungroup()

# View the new data frame with the genre columns, as well as the updated edx
head(edx2)
genres

#Data Visualization------------------------------------------------------------

#Histogram of frequency for each possible rating (1 to 5)

edx %>% ggplot(aes(x = rating)) +
  geom_histogram(binwidth = 0.5, fill = "steelblue", color = "white") +
  labs(title = "Distribution of Ratings", x = "Ratings", y = "Frequency")

#Histogram of frequency for the number of ratings

ggplot(data = edx, aes(x = num_ratings)) +
  geom_histogram(binwidth = 500) +
  xlab("Number of Ratings") +
  ylab("Frequency") +
  ggtitle("Distribution of Number of Ratings")


#mean rating line vs year - showing years variability. 
#Useful to explore effect of year on rating.
edx %>% group_by(year) %>%
  summarise(mean_rating = mean(rating), sd_rating = sd(rating)) %>%
  ggplot(aes(x = year, y = mean_rating)) +
  geom_line() +
  geom_ribbon(aes(ymin = mean_rating - sd_rating, ymax = mean_rating + sd_rating), alpha = 0.2) +
  labs(title = "Mean Rating by Year with Standard Deviation") +
  xlab("Year") +
  ylab("Mean Rating")

#creating 20 different clusters for date and the study it's effect 
#on rating. We do this by plotting on a geom_boxplot format.

edx %>% 
  mutate(date = round_date(date, unit = "year")) %>%
  mutate(cluster = cut(date, breaks = 6)) %>% #Create 20 clusters of date
  group_by(cluster) %>%
  mutate(cluster_order = min(date)) %>% # get the earliest date in each cluster
  ungroup() %>% 
  arrange(cluster_order) %>% # sort the clusters based on the earliest date in each cluster
  ggplot(aes(x = cluster, y = rating)) +
  geom_boxplot(fill = "light blue", alpha = 0.3, outlier.alpha = 0) + 
  labs(x = "Date Cluster", y = "Rating", title = "Mean Rating by Date of Rating - 6 Clusters") +
  scale_x_discrete(labels = function(x) gsub("\\..*", "", as.character(x)))

#plot of movieID vs ratings. We start by summarizing (mean rate) 
#by MovieID. Then we only keep movies with 100+ ratings.

edx %>%
  group_by(movieId) %>%
  summarise(avg_rating = mean(rating),
            num_ratings = n()) %>%
  filter(num_ratings > 100) %>% # Only keep movies with more than 50 ratings
  ggplot(aes(x = avg_rating, y = num_ratings, size = num_ratings)) +
  geom_point(alpha = 0.2) +    # we do some trials to find the right alpha and max_size
  scale_size_area(max_size = 8) +
  labs(x = "Average Rating", y = "Number of Ratings", title = "Mean Rating by MovieId (Including Num of ratings)") +
  theme_classic()

#Similarly, we plot of userID vs rating. We start by summarizing (mean rate) by userID.
edx %>% group_by(userId) %>%
  summarise(avg_rating = mean(rating),
            num_ratings = n()) %>%
  filter(num_ratings > 50)  %>% # Only keep users with more than 50 ratings
  ggplot(aes(x = avg_rating, y = num_ratings, size = num_ratings)) +
  geom_point(alpha = 0.2) +    
  scale_size_area(max_size = 8) +
  labs(x = "Average Rating", y = "Number of Ratings", title = "Mean Rating by UserId (Including Num of ratings)") +
  theme_classic()

#summarize the average rating per genre and analyze it

# create a data frame with the average rating and count per genre
genre_data <- edx2 %>%
  select(-c(userId, movieId, timestamp, title, year, num_ratings, date)) %>%
  pivot_longer(cols = -rating, names_to = "genre", values_to = "has_genre") %>%
  filter(has_genre == 1) %>%
  select(-has_genre)

genre_summary <- genre_data %>%
  group_by(genre) %>% 
  summarise(avg_rating = mean(rating),
            var_rating = var(rating),
            count = n()) %>%  filter(count >= 1000) %>% 
  mutate(genre = reorder(genre, avg_rating))

# Plot the mean rating and variance for each genre
ggplot(genre_summary, aes(x = genre, 
                          y = avg_rating, ymin = avg_rating - var_rating, ymax = avg_rating + var_rating)) +
  geom_col(fill = "lightblue", width = 0.8, position = position_dodge()) +
  geom_errorbar(width = 0.2, position = position_dodge(0.8)) +
  labs(x = "Genre", y = "Average Rating", title = "Mean Rating by Genre") +
  coord_flip() +
  theme_minimal()

##Alternatively, we can do a similar exercise understanding "genres" as 2500+ unique factors

edx %>% 
  group_by(genres) %>%
  summarize(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n())) %>%
  filter(n >= 50000) %>% 
  mutate(genres = reorder(genres, avg)) %>%
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Genres", y = "Average Rating", 
       title = "Average Rating by Genre Combination")

#Machine Learning Models creation-------------------------------------------

#We start by dividing our data-set into train and test data.
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
edx_train <- edx[-test_index,] #train data
edx_test <- edx[test_index,] #test data

global_mean <- mean(edx_train$rating) #estimate the global mean of train data
n <- nrow(edx_test)
predicted_ratings_onlyMean <- rep(global_mean, n) #all predictions are the global mean
actual_ratings <- edx_test$rating
rmse_onlyMean <- RMSE(predicted_ratings_onlyMean, actual_ratings)
print(rmse_onlyMean)

# Calculate movie biases
movie_biases <- edx_train %>%
  group_by(movieId) %>%
  summarize(movie_bias = mean(rating - global_mean))

# Make predictions on test data using MovieId bias correction
test_predictions_movieBias <- edx_test %>%
  left_join(movie_biases, by = "movieId") %>%
  mutate(movie_bias = if_else(is.na(movie_bias), 0, movie_bias),
         predicted_rating = global_mean + movie_bias)

rmse_movieBias <- RMSE(edx_test$rating, test_predictions_movieBias$predicted_rating)
print(rmse_movieBias)

#Calculate user bias following the previous approach
user_biases <- edx %>%
  group_by(userId) %>%
  summarize(user_bias = mean(rating - global_mean))
# Make predictions on test data using MovieId bias correction
test_predictions_UserMovie <- edx_test %>%
  left_join(movie_biases, by = "movieId") %>%
  left_join(user_biases, by = "userId") %>%
  mutate(movie_bias = if_else(is.na(movie_bias), 0, movie_bias),
         user_bias = if_else(is.na(user_bias), 0, user_bias),
         predicted_rating = global_mean + movie_bias + user_bias)

rmse_movieUserBias <- RMSE(edx_test$rating, test_predictions_UserMovie$predicted_rating)
print(rmse_movieUserBias)

# Create a function to calculate movie biases with regularization
movie_biases_regularized <- function(train_data, lambda) {
  train_data %>%
    group_by(movieId) %>%
    summarize(movie_bias = sum(rating - global_mean) / (n() + lambda))
}

# Define the range of lambda values to test
lambda_seq <- seq(0.1, 20, by = 0.1)

# Initialize an empty data frame to store lambda and corresponding RMSE values
lambda_rmse <- data.frame(lambda = numeric(), rmse = numeric())

# Loop through the lambda values
for (lambda in lambda_seq) {
  # Calculate movie biases with regularization
  movie_biases_reg <- movie_biases_regularized(edx_train, lambda)
  
  # Make predictions on test data
  test_predictions_regul <- edx_test %>%
    left_join(movie_biases_reg, by = "movieId") %>%
    left_join(user_biases, by = "userId") %>%
    mutate(movie_bias = if_else(is.na(movie_bias), 0, movie_bias),
           user_bias = if_else(is.na(user_bias), 0, user_bias),
           predicted_rating = global_mean + movie_bias + user_bias)
}

rmse_regul <- RMSE(edx_test$rating, test_predictions_regul$predicted_rating)
print(rmse_regul)

#genre bias calculation
min_genre_ratings <- 2000  # Set the minimum number of ratings for a 
#genre combination to avoid over fitting.

# Calculate genre biases --------------------------------------------------------
genre_biases <- edx2 %>%
  select(c("userId", "movieId", "rating", genres)) %>%
  gather("genre", "genre_present", genres) %>%
  filter(genre_present == 1) %>%
  group_by(genre) %>%
  filter(n() >= min_genre_ratings) %>%
  summarize(genre_bias = mean(rating - global_mean)) %>%
  select(all_of("genre"), genre_bias)

#this function takes each value of the "genres" column and breaks it 
#into a genres vector. Then estimates the sum of the genres vector #multiplied for the genres biases for each test observation

calculate_total_genre_bias <- function(genres_str, genre_biases_df) {
  genre_list <- strsplit(genres_str, "\\|")[[1]]
  genre_bias_values <- genre_biases_df$genre_bias[genre_biases_df$genre %in% genre_list]
  sum(genre_bias_values, na.rm = TRUE)
}
#follow the same approach to calculate the prediction. 
#But this time we make use of the above function to estimate each genre bias.

test_predictions_withGenreBias <- edx_test %>%
  left_join(movie_biases_reg, by = "movieId") %>%
  left_join(user_biases, by = "userId") %>%
  mutate(movie_bias = if_else(is.na(movie_bias), 0, movie_bias),
         user_bias = if_else(is.na(user_bias), 0, user_bias),
         genre_bias = map_dbl(genres, calculate_total_genre_bias, genre_biases),
         predicted_rating = global_mean + movie_bias + user_bias + genre_bias)

rmse_genre <- RMSE(edx_test$rating, test_predictions_withGenreBias$predicted_rating)
print(rmse_genre)

#First we create the rating matrix (necessary input for using the #recommender standard function). We filter the data to improve #computing time.

min_ratings <- 100

filtered_data <- edx %>%
  group_by(movieId) %>%
  filter(n() >= min_ratings) %>%
  ungroup() %>%
  group_by(userId) %>%
  filter(n() >= min_ratings) %>%
  ungroup()


y <- select(filtered_data, movieId, userId, rating) %>%
  pivot_wider(names_from = movieId, values_from = rating) 
y <- as.matrix(y[,-1])

colnames(y) <- filtered_data %>% select(movieId, title) %>% 
  distinct(movieId, .keep_all = TRUE) %>%
  right_join(data.frame(movieId=as.integer(colnames(y))), by = "movieId") %>%
  pull(title)

ratings_matrix <- as(y, "realRatingMatrix")

#We take standard parameters provided on literature.
given <- 15
n_folds <- 15

#translate the rating matrix in a format that can be used by the #Recommender function
evaluation_scheme <- evaluationScheme(
  ratings_matrix, 
  method='split',
  train= 0.9,
  k=n_folds,
  given= given)



ratings_train <- getData(evaluation_scheme, 'train')

ratings_test_known <- getData(evaluation_scheme, 'known')

ratings_test_unknown <- getData(evaluation_scheme, 'unknown')
#We proceed to train our SVD model with the rating matrix above.

SVD_model <- Recommender(
  data=ratings_train,
  method='SVD')
#Finally, we apply the model using the standard function predict()

svd_prediction <- predict(object = SVD_model, newdata = ratings_test_known, n = 10, type = "ratings")
#Now we calculate the accuracy of the model using the "unkown" data.
rmse_sdv <- calcPredictionAccuracy(x = svd_prediction, data = ratings_test_unknown, byUser = FALSE, measure = "RMSE")

rmse_sdv <- as.numeric(rmse_sdv[1])
print(rmse_sdv)

# Convert the train and test sets into recosystem input format
train_data <- with(edx_train, data_memory(user_index = userId, item_index = movieId, rating = rating))
test_data <- with(edx_test, data_memory(user_index = userId, item_index = movieId, rating = rating))

# Create the model object
r <- recosystem::Reco()

# Select the best tuning parameters
opts <- r$tune(train_data,
               opts = list(dim = c(10, 20, 30),          
                           # dim is number of factors
                           lrate = c(0.1, 0.2),         
                           # learning rate
                           costp_l2 = c(0.01, 0.1),      #regularization for P factors
                           costq_l2 = c(0.01, 0.1),      #regularization for Q factors
                           nthread = 4, niter = 10))     # convergence controlled by number of iterations and learning rate

# Train the algorithm
r$train(train_data, opts = c(opts$min, nthread = 4, niter = 20))

# Predict the ratings for the test data
predicted_ratings <- r$predict(test_data)


# You can calculate the RMSE or other evaluation metrics using the true #and predicted ratings
rmse_reco <- RMSE(edx_test$rating, predicted_ratings)


#Results-----------------------------------------------------------------------

Models_Performance <- data.frame(
  Model = c(
    "Global Mean as Prediction",
    "Movie Bias",
    "Movie and User Bias",
    "Regularized Movie Bias + User Bias",
    "SVD Model",
    "Recosystem Model"
  ),
  RMSE = c(
    rmse_onlyMean,
    rmse_movieBias,
    rmse_movieUserBias,
    rmse_regul,
    rmse_sdv,
    rmse_reco
  )
)

# Print the results
print(Models_Performance)

# Convert the train and test sets into recosystem input format
train_data <- with(edx, data_memory(user_index = userId, item_index = movieId, rating = rating))
test_data <- with(final_holdout_test, data_memory(user_index = userId, item_index = movieId, rating = rating))

# Create the model object
r <- recosystem::Reco()

#Please notice that we avoid the calculation again for the optimal #parameters due to computational timing (30 min) and the similarity #between the two train data edx_train and edx as a whole.

r$train(train_data, opts = c(opts$min, nthread = 4, niter = 20))

# Predict the ratings for the test data
predicted_ratings <- r$predict(test_data)


# You can calculate the RMSE or other evaluation metrics using the true and predicted ratings
rmse_final_prediction <- RMSE(final_holdout_test$rating, predicted_ratings)


# Add the final prediction to the Models_Performance dataframe
Models_Performance <- rbind(Models_Performance,
                            data.frame(Model = "** Recosystem Model - Final **",
                                       RMSE = rmse_final_prediction))

# Print the results
print(Models_Performance)


#Thanks!