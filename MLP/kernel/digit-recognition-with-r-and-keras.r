
## Load Libraries
library(dplyr)
library(magrittr)
library(keras)
library(tictoc) #For timing

## Path to the data files
data_file.path <- "~/Kaggle/DigitRecoginizer/data"
data_file.train <- paste(data_file.path, "train.csv", sep="/")
data_file.test <- paste(data_file.path, "test.csv", sep="/")

## Load Data
digits.train <- read.csv(data_file.train)
digits.test <- read.csv(data_file.test)

barplot(table(digits.train[, 1]), col = "blue", main = "Count of Digits in Training Set")

##Scale the pixel data
pixels.train <- digits.train[,-1]/255


##Scaled training set
train.scaled <- cbind('label' = digits.train$label,
                      pixels.train)

##Display a 10x12 matrix of images
par(mfrow=c(10, 12), pty='s', mai=c(0.2, 0, 0, 0.1))

for (lab in 0:9) {
  samp <- train.scaled %>%
    filter(label == lab)
  for (i in 1:12) {
    img <- matrix(as.numeric(samp[i, -1]), 28, 28, byrow = TRUE)
    image(t(img)[,28:1], axes = FALSE, col = grey(seq(1, 0, length = 256)))
    box(lty = 'solid')
  }
}

par(mfrow=c(2, 5), pty='s', mai=c(0, 0, 0, 0))

for (lab in 0:9) {
  subs <- train.scaled %>%
    filter(label == lab)
  avg <- colMeans(subs)
  img <- matrix(avg[2:length(avg)], 28, 28, byrow = TRUE)
  image(t(img)[,28:1], axes = FALSE, col = grey(seq(1, 0, length = 256)))
  box(lty = 'solid')
}

one_hot_labels <- to_categorical(digits.train$label, 10)
dimnames(one_hot_labels) <- list(NULL, c(0:9))


test_size <- 1000
val_size <- .1 * nrow(digits.train)

pixels <- digits.train[, -1] %>%
    split(rep(c('test', 'val', 'train'), 
              c(test_size, val_size, nrow(digits.train) - test_size - val_size))) %>%
    sapply(as.matrix)

labels <- digits.train[, 1] %>%
    split(rep(c('test', 'val', 'train'), 
            c(test_size, val_size, nrow(digits.train) - test_size - val_size))) %>%
    sapply(to_categorical, 10) 

lapply(labels, function (x) colnames(x) <- c(0:9))


NN <- keras_model_sequential()
NN %>%
    layer_dense(units = 128, activation = "relu", input_shape = c(784)) %>%
    layer_dense(units = 10, activation = "softmax") 
    
NN %>%
    compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = "accuracy")

tic("Fitting Model")
history <- NN %>% 
    fit(pixels$train, one_hot_labels, 
        epochs = 30, 
        validation_split = 0.1, 
        batch_size = 64, 
        verbose = 2)
toc()

plot(history)
