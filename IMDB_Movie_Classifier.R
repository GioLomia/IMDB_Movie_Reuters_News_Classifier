library(keras)
library(xgboost)
library(magrittr)
library(dplyr)
library(Matrix)
library(tidyselect)
library(keras)
library(tidyverse)
library(recipes)
library(ROCR)
library(mlbench)
library(DataExplorer)
library(tidyverse)
library(polycor)
library(car)
library(broom)
library(dplyr)
library(rsample)
library(class)
library(caret)
library(ROSE)
library(randomForest)
library(glmnet)
library(gbm)

######################IMDB##################
max_features <- 15000                                         
maxlen <- 1500                                                
imdb <- dataset_imdb(num_words = max_features)
c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb
glimpse(x_train)

x_train <- pad_sequences(x_train, maxlen = maxlen)            
x_test <- pad_sequences(x_test, maxlen = maxlen)
glimpse(x_train)
dim(x_train)
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 15000, output_dim = 5,input_length = maxlen) %>%
  layer_lstm(units = 10)%>%
  layer_flatten() %>%   
  layer_dropout(rate=.3)%>%
  layer_dense(units=100,activation="relu",regularizer_l1_l2())%>%
  layer_dropout(rate=.40)%>%
  layer_dense(units = 20,activation = "relu")%>%
  layer_dense(units = 46, activation = "softmax")                

model

model %>% compile(
  optimizer = "nadam",
  loss = "categorical_crossentropy",
  metrics = c("acc")
)

summary(model)
history <- model %>% fit(
  x_train, as.matrix(y_train),
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)




########################Reuters#######################
max_features <- 15000                                         
maxlen <- 200                                                
reu <- dataset_reuters(num_words = max_features)
c(c(x_train, y_train), c(x_test, y_test)) %<-% reu
glimpse(y_train)

y_train<-to_categorical(y_train)
x_train <- pad_sequences(x_train, maxlen = maxlen)            
x_test <- pad_sequences(x_test, maxlen = maxlen)
glimpse(x_train)


model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 15000, output_dim = 5,            
                  input_length = maxlen) %>%
  layer_conv_1d(10,5,activation = "relu")%>%
  layer_max_pooling_1d()%>%
  layer_conv_1d(16,5,activation = "relu")%>%
  layer_max_pooling_1d()%>% 
  layer_flatten() %>% 
  layer_dense(units = 64,activation = "relu")%>%
  layer_dense(units = 46, activation = "softmax")                

model

model %>% compile(
  optimizer = "nadam",
  loss = "categorical_crossentropy",
  metrics = c("acc")
)

summary(model)
history <- model %>% fit(
  as.matrix(x_train), y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)


