library(neuralnet)
library(nnet)
library(tictoc)

#data_file.path <- "../input"
data_file.path <- "/home/viet/Kaggle/DigitRecoginizer/data"
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
accuracy <- NULL
weights <- NULL
K = 10
tic("Total Training Time")
for (k in 1:K) {
    hold <- (.10*(k-1)*nrow(training.scaled) + 1):(.10*k*nrow(training.scaled))
    tr <- training.scaled[-hold, ]
    vl <- training.scaled[hold, ]
    tic(paste("Epoch", k))
    print(paste("Starting Epoch", k))
    nn.model <- neuralnet(f, data = tr, hidden = c(floor(length(pixels.select) * 2/3)), linear.output = FALSE)
    toc()
    nn.predict <- compute(nn.model, vl[, pixels.select])

    accuracy[k] <- sum(max.col(nn.predict$net.result) - 1 == digits.train$label[hold])/(.10 * nrow(digits.train))
    print(accuracy)
    print(paste("Mean Accuracy:", mean(accuracy)))
    if (k == 1 || (accuracy[k] > accuracy[k-1])) {
        nn.model.best <- nn.model
    }
}
toc()

te <- digits.test/255
nn.test <- compute(nn.model.best, te[, pixels.select])

sub <- data.frame(ImageId = 1:28000, Label = max.col(nn.test$net.result) - 1)
write.csv(sub, file = "submission.csv", row.names = FALSE, quote = FALSE)