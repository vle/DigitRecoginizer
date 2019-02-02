library(neuralnet)
library(nnet)
library(tictoc)

data_file.path <- "../data"
data_file.train <- paste(data_file.path, "train.csv", sep="/")
data_file.test <- paste(data_file.path, "test.csv", sep="/")

print(paste("model started ", Sys.time()))

tic.clear()

tic("prepping data")
digits.train <- read.csv(data_file.train)
digits.test <- read.csv(data_file.test)

onehot_labels <- class.ind(as.factor(digits.train$label))
onehot_labels.names <- sapply(0:9, function (x) paste("label", x, sep = ""))
training.scaled <- cbind(onehot_labels, digits.train[,-1]/255)
names(training.scaled) <- c(onehot_labels.names, names(training.scaled)[11:794])


pixels.sum <- colSums(training.scaled[,11:794])
pixels.norm <- pixels.sum/max(pixels.sum)

pixels.select <- names(pixels.norm[pixels.norm > 0.05])
toc()

f <- as.formula(paste(paste(onehot_labels.names, collapse = '+'), paste(pixels.select, collapse = '+'), sep = '~'))

# training.data <- training.scaled[1:4200,]
# training.validation <- training.scaled[4201:8400,]
# tic("Total")
# tic("epoch 1")
# print("epoch 1")
# nn.model <- neuralnet(f, data = training.data, hidden = c(128, 64), linear.output = FALSE)
# nn.predict <- compute(nn.model, training.validation[, pixels.select])
# accuracy <- NULL
# accuracy[1] <- sum((max.col(nn.predict$net.result) - 1) == digits.train$label[4201:8400])/4200
# print(accuracy)
# toc(func.toc = function(tic, toc, msg) outmsg <- paste(msg, "done: ", round(round(toc - tic, 3)/60, 2), "minutes"))
# 
# K = 10
# for (k in 2:(K-1)) {
#   tic(paste("epoch", k))
#   print(paste("epoch", k, Sys.time()))
#   training.data <- training.scaled[(1 + (k-1)*4200):(k*4200), ]
#   training.validation <- training.scaled[(1 + k*4200):((k+1)*4200), ]
#   nn.model <- neuralnet(f, data=training.data, hidden = c(128, 64), linear.output = FALSE, startweights = nn.model$weights)
#   nn.predict <- compute(nn.model, training.validation[, pixels.select])
#   accuracy[k] <- sum((max.col(nn.predict$net.result) - 1) == digits.train$label[(1 + k*4200):((k+1)*4200)])/4200
#   print(accuracy)
#   toc(func.toc = function(tic, toc, msg) outmsg <- paste(msg, "done: ", round(round(toc - tic, 3)/60, 2), "minutes"))
# }
# 
# k=10
# tic("epoch 10")
# print(paste("epoch 10", Sys.time()))
# training.data <- training.scaled[(1 + (k-1)*4200):(k*4200), ]
# training.validation <- training.scaled[1:4200, ]
# nn.model <- neuralnet(f, data=training.data, hidden = c(128, 64), linear.output = FALSE, startweights = nn.model$weights)
# nn.predict <- compute(nn.model, training.validation[, pixels.select])
# accuracy[k] <- sum((max.col(nn.predict$net.result) - 1) == digits.train$label[1:4200])/4200
# print(accuracy)
# toc(func.toc = function(tic, toc, msg) outmsg <- paste(msg, "done: ", round(round(toc - tic, 3)/60, 2), "minutes"))
# toc(func.toc = function(tic, toc, msg) outmsg <- paste(msg, "done: ", round(round(toc - tic, 3)/60, 2), "minutes"))

index <- sample(1:42000, 0.9 * 42000)
tr <- training.scaled[index, ]
vl <- training.scaled[-index, ]
tic("training model")
nn.model <- neuralnet(f, data = tr, hidden = c(128, 64), linear.output = FALSE, lifesign = 'full')
toc()
nn.predict <- compute(nn.model, vl[, pixels.select])

te <- digits.test/255

nn.test <- compute(nn.model, te[, pixels.select])

sub <- data.frame(ImageId = 1:28000, Label = max.col(nn.test$net.result))
write.csv(sub, file = "submission.csv", row.names = FALSE, quote = FALSE)
