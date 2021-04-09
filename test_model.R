# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.


# biblioteki 
library(mlflow)
library(glmnet)
library(carrier)

# deklaracja zmiennej srodowiskowej
Sys.setenv(MLFLOW_S3_ENDPOINT_URL="http://20.52.130.217:9000")
Sys.setenv(MLFLOW_TRACKING_URI = "http://20.52.130.217:5000")
# remote_server_uri = "http://20.52.130.217:5000" # set to your server URI

# mlflow_set_tracking_uri(remote_server_uri)

# deklaracja nazwy eksperymentu w ramach ktorego beda przeprowadzane eksperymenty
mlflow_set_experiment("/test_brite2")

# Sys.setenv(AWS_ACCESS_KEY_ID="klucz")
# Sys.setenv(AWS_SECRET_ACCESS_KEY="klucz2")
set.seed(40)

# Read the wine-quality csv file
data <- read.csv("1. Dane/wine-quality.csv")
set.seed(40)

# Split the data into training and test sets. (0.75, 0.25) split.
sampled <- sample(1:nrow(data), 0.75 * nrow(data))
train <- data[sampled, ]
test <- data[-sampled, ]

# The predicted column is "quality" which is a scalar from [3, 9]
train_x <- as.matrix(train[, !(names(train) == "quality")])
test_x <- as.matrix(test[, !(names(train) == "quality")])
train_y <- train[, "quality"]
test_y <- test[, "quality"]

# deklaracja parametrow modelu
alpha <- mlflow_param("alpha", 0.5, "numeric")
lambda <- mlflow_param("lambda", 0.5, "numeric")

# Wykonanie ranu w ramach funkcji with, po zakonczeniu funkcji run zostanie automatycznie zamkniety
with(mlflow_start_run(), {
    model <- glmnet(train_x, train_y, alpha = alpha, lambda = lambda, family= "gaussian", standardize = FALSE)
    
    # izolowanie funkcji predykcyjne, funkcja crate z pakietu carrier
    predictor <- crate(~ glmnet::predict.glmnet(!!model, as.matrix(.x)), !!model)
    
    # predykcja wartosci
    predicted <- predictor(test_x)

    # obliczanie metryk jakosci
    rmse <- sqrt(mean((predicted - test_y) ^ 2))
    mae <- mean(abs(predicted - test_y))
    r2 <- as.numeric(cor(predicted, test_y) ^ 2)

    # komunikaty przekzywane do konsoli
    message("Elasticnet model (alpha=", alpha, ", lambda=", lambda, "):")
    message("  RMSE: ", rmse)
    message("  MAE: ", mae)
    message("  R2: ", r2)

    # zapisywanie parametrow uzytych w modelu
    mlflow_log_param("alpha", mlflow_param("alpha", 0.7))
    mlflow_log_param("lambda", mlflow_param("lambda", 0.8))
    
    for(i in c(1:3)){
      mlflow_log_metric("moja metryka", i)
    }
    
    # metryki zapisywane przez mlflow
    mlflow_log_metric("rmse", rmse)
    mlflow_log_metric("r2", r2)
    mlflow_log_metric("mae", mae)

    # zapisywanie artefaktu model
    mlflow_log_model(predictor, "model")

    # zapisywanie artefaktu, dane do treningu. Artefakty beda dostepne w ui mlflow
    write.csv(train_x, "train_x.csv")
    mlflow_log_artifact("train_x.csv")
})

