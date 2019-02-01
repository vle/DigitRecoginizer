library(neuralnet)

data_file.path <- "../data"
data_file.train <- paste(data_file.path, "train.csv", sep="/")
data_file.test <- paste(data_file.path, "test.csv", sep="/")

digits.train <- read.csv(data_file.train)
digits.test <- read.csv(data_file.test)

training.scaled <- cbind('label' = digits.train$label, digits.train[,-1]/255)

pixels.sum <- colSums(training.scaled[,-1])
pixels.norm <- pixels.sum/max(pixels.sum)

pixels.select <- pixels.norm[pixels.norm > 0.05]

f <- as.formula(paste("label~", paste(names(pixels.select), collapse = "+")))

training.data <- training.scaled[1:4200,]
training.validation <- training.scaled[4201:8400,]

#nn.model <- neuralnet(f, data = training.data, hidden = c(128, 64))
#nn.predict <- compute(nn.model, training.validation[, -1])