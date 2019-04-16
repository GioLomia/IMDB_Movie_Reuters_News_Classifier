library(keras)

library(tidyverse)

max_features <- 10000
maxlen <- 500

imbd <- dataset_imdb(num_words = max_features)
c(c(x_train, y_train), c(x_test, y_test)) %<-% imbd
x_train <- pad_sequences(x_train, maxlen = maxlen)
x_test <- pad_sequences(x_test, maxlen = maxlen)

################STEP 1###################
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = 32,
                  input_length = maxlen) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

#################STEP 2##################
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = 32,
                  input_length = maxlen) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

##################STEP 3###################
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 10000, output_dim = 32,
                  input_length = maxlen) %>%
  layer_lstm(units = 32)%>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

#################STEP 4###############
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 10000, output_dim = 32,
                  input_length = maxlen) %>%
  layer_lstm(units = 32,recurrent_dropout = 0.2,dropout = 0.3)%>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

#################STEP 5#################
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = 32) %>%
  bidirectional( layer_lstm(units = 32,recurrent_dropout = 0.2,dropout = 0.3) )%>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

#################STEP 6#################
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = 8) %>%
  layer_lstm(units = 8,recurrent_dropout = 0.2,dropout = 0.3,return_sequences = TRUE)%>%
  layer_lstm(units = 8,recurrent_dropout = 0.2,dropout = 0.3)%>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

#################FINAL MODEL###########################
model2 <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features,
                  output_dim = 8,
                  input_length = maxlen) %>%
  layer_gru(units = 18,activation = "relu",return_sequences = T) %>%
  layer_gru(units = 28,activation = "relu",return_sequences = T) %>%
  layer_gru(units = 38,activation = "relu",recurrent_dropout = 0.01,return_sequences = T) %>%
  layer_gru(units = 48,activation = "relu") %>%
  layer_dense(units = 64, activation = "relu")%>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)


history <- model %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)

model

plot(history)



##############################REUTERS#######################

max_features <- 10000
maxlen <- 500

reuters <- dataset_reuters(num_words = max_features)
c(c(x_train, y_train), c(x_test, y_test)) %<-% reuters
x_train <- pad_sequences(x_train, maxlen = maxlen)
x_test <- pad_sequences(x_test, maxlen = maxlen)
y_train<-to_categorical(y_train)
################STEP 1###################
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = 32,
                  input_length = maxlen) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 46, activation = "softmax")

#################STEP 2##################
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = 32,
                  input_length = maxlen) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 46, activation = "softmax")

##################STEP 3###################
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 10000, output_dim = 32,
                  input_length = maxlen) %>%
  layer_lstm(units = 32)%>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 46, activation = "softmax")

#################STEP 4###############
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 10000, output_dim = 32,
                  input_length = maxlen) %>%
  layer_lstm(units = 32,recurrent_dropout = 0.2,dropout = 0.3)%>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 46, activation = "softmax")

#################STEP 5#################
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = 32) %>%
  bidirectional( layer_lstm(units = 32,recurrent_dropout = 0.2,dropout = 0.3) )%>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 46, activation = "softmax")

#################STEP 6#################
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = 8) %>%
  layer_lstm(units = 8,recurrent_dropout = 0.2,dropout = 0.3,return_sequences = TRUE)%>%
  layer_lstm(units = 8,recurrent_dropout = 0.2,dropout = 0.3)%>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 46, activation = "softmax")

model

#################FINAL MODEL###########################
model2 <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features,
                  output_dim = 8,
                  input_length = maxlen) %>%
  layer_gru(units = 18,activation = "relu",return_sequences = T) %>%
  layer_gru(units = 28,activation = "relu",return_sequences = T) %>%
  layer_gru(units = 38,activation = "relu",recurrent_dropout = 0.01,return_sequences = T) %>%
  layer_gru(units = 48,activation = "relu") %>%
  layer_dense(units = 64, activation = "relu")%>%
  layer_dense(units = 46, activation = "softmax")



model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("acc")
)


history <- model %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)

model

plot(history)

