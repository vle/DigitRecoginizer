library(keras)
library(tictoc)

data_file.path <- "~/Kaggle/DigitRecoginizer/data"
data_file.train <- paste(data_file.path, "train.csv", sep="/")
data_file.test <- paste(data_file.path, "test.csv", sep="/")

digits.train <- read.csv(data_file.train)
digits.test <- read.csv(data_file.test)

one_hot_labels <- to_categorical(digits.train$label, 10)
dimnames(one_hot_labels) <- list(NULL, c(0:9))
pixels.train <- as.matrix(digits.train[,-1]/255)
pixels.test <- as.matrix(digits.test/255)

val_split = 0.1
hold <- sample(1:nrow(pixels.train), val_split * nrow(pixels.train))
pixels.val <- pixels.train[hold, ]
labels.val <- one_hot_labels[hold, ]
pixels.train <- pixels.train[-hold, ]
labels.train <- one_hot_labels[-hold, ]

dim(pixels.train) <- c(nrow(pixels.train), 28, 28, 1)
dim(pixels.val) <- c(nrow(pixels.val), 28, 28, 1)
dim(pixels.test) <- c(nrow(pixels.test), 28, 28, 1)

CNN <- keras_model_sequential()

CNN %>%
    layer_conv_2d(filters = 32, kernel_size = 3, activation = "relu", input_shape = c(28, 28, 1)) %>%
    layer_conv_2d(filters = 32, kernel_size = 3, activation = "relu") %>%
    layer_max_pooling_2d() %>%
    layer_dropout(rate = 0.25) %>%
    layer_conv_2d(filters = 64, kernel_size = 3, activation = "relu") %>%
    layer_conv_2d(filters = 64, kernel_size = 3, activation = "relu") %>%
    layer_max_pooling_2d() %>%
    layer_dropout(rate = 0.25) %>%
    layer_flatten() %>%
    layer_dense(units = 128, activation = "relu") %>%
    layer_dropout(rate = 0.50) %>%
    layer_dense(units = 10, activation = "softmax")
    
CNN %>%
    compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = "accuracy")

tic.clear()
batch_size = 64
early_stop <- callback_early_stopping(monitor = "val_loss",
                                      patience = 5,
                                      mode = "min",
                                      restore_best_weights = TRUE,
                                      verbose = 1)
reduce_lr <- callback_reduce_lr_on_plateau(monitor = "val_acc", 
                                           patience = 3,
                                           factor = 0.5,
                                           mode = "max",
                                           verbose = 1)
#tic("Fitting Model")
#history <- CNN %>% 
#    fit(pixels.train, labels.train, 
#        epochs = 30, 
#        validation_data = list(pixels.val, labels.val), 
#        batch_size = batch_size, 
#        verbose = 2)
#toc()



img_gen <- image_data_generator(rotation_range = 30, 
                                width_shift_range = 0.15, 
                                height_shift_range = 0.15,
                                zoom_range = 0.05)

tic("Fitting Model with Augmented Data")
history <- CNN %>%
    fit_generator(flow_images_from_data(pixels.train, labels.train, 
                                        generator = img_gen,
                                        batch_size = batch_size),
                    steps_per_epoch = floor(nrow(pixels.train)/batch_size),
                    epochs = 30,
                    validation_data = list(pixels.val, labels.val),
                    callbacks = list(reduce_lr, early_stop),
                    verbose = 2)
toc()

plot(history)

test.labels <- CNN %>%
    predict_classes(pixels.test, batch_size = batch_size, verbose = 0)
    
sub <- data.frame(ImageId = 1:28000, Label = test.labels)
write.csv(sub, file = "submission.csv", row.names = FALSE, quote = FALSE)