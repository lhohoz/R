library(data.table)
library(xgboost)
library(Matrix)
library(Metrics)
library(car)
library(e1071)
library(caret)
library(Hmisc)
library(forecast)


train <- fread("/Users/usagi/Desktop/kaggle/allstate/train.csv")
test <- fread("/Users/usagi/Desktop/kaggle/allstate/test.csv")

train[, logloss:= log(loss + 200)]
Y <- train$logloss
train.id <- train$id
test.id <- test$id
train[, c('id', 'loss', 'logloss') := NULL]
test[, id := NULL]
row.train <- nrow(train)

COMB_FEATURE <- c('cat80','cat87','cat57','cat12','cat79','cat10','cat7','cat89',
                  'cat2','cat72','cat81','cat11','cat1','cat13','cat9','cat3','cat16',
                  'cat90','cat23','cat36','cat73','cat103','cat40','cat28','cat111',
                  'cat6','cat76','cat50','cat5','cat4','cat14','cat38','cat24','cat82','cat25')

comb_features <- paste(combn(COMB_FEATURE, 2)[1,], combn(COMB_FEATURE, 2)[2,], sep="_")
cat_features <- names(train)[grep("cat", names(train))]
cont_features <- names(train)[grep("cont", names(train))]

train_unique <- sapply(train[, c(cat_features), with=FALSE], function(x){unique(x)})
test_unique <- sapply(test[, c(cat_features), with=FALSE], function(x){unique(x)})

equal <- function(x, y){
  if (length(x) != length(y)){
    set_train <- x
    set_test <- y
    remove_train <- setdiff(set_train, set_test)
    remove_test <- setdiff(set_test, set_train)
    remove <- union(remove_train, remove_test)
  }
}
remove_list <- mapply(equal, train_unique, test_unique)
cat_dt_train <- data.table(apply(train[, c(cat_features), with=FALSE], 2, function(x){ifelse(x %in% remove_list, NA, x)}))
train <- cbind(cat_dt_train, train[, -c(cat_features), with=FALSE])
rm(cat_dt_train)

cat_dt_test <- data.table(apply(test[, c(cat_features), with=FALSE], 2, function(x){ifelse(x %in% remove_list, NA, x)}))
test <- cbind(cat_dt_test, test[, -c(cat_features), with=FALSE])
rm(cat_dt_test)

for(var in cat_features) {
  levels <- unique(c(train[[var]], test[[var]]))
  set(train, j = var, value = factor(train[[var]], levels = levels))
  set(test, j = var, value = factor(test[[var]], levels = levels))
}

all_data <- rbind(train, test)
all_data$i <- 1:dim(all_data)[1]

for (comb_fea in comb_features){
  first_cat <- strsplit(comb_fea,"_")[[1]][1]
  second_cat <- strsplit(comb_fea,"_")[[1]][2]
  all_data[, c(comb_fea) := list(paste0(all_data[[first_cat]], all_data[[second_cat]]))]
}

for(var in comb_features) {
  levels <- unique(all_data[[var]])
  set(all_data, j = var, value = factor(all_data[[var]], levels = levels))
}

#for (f in cont_features) {
#  tst <- e1071::skewness(all_data[, eval(as.name(f))])
#  if (tst > .25) {
#    if (is.na(all_data[, BoxCoxTrans(eval(as.name(f)))$lambda])) next
#    all_data[, eval(as.name(f)) := BoxCox(eval(as.name(f)), BoxCoxTrans(eval(as.name(f)))$lambda)]
#  }
#}

all_data[, (cat_features) := lapply(.SD, as.numeric), .SDcols=cat_features]
all_data[, (comb_features) := lapply(.SD, as.numeric), .SDcols=comb_features]
all_data[, (cont_features) := lapply(.SD, scale), .SDcols=cont_features]
all_data <- all_data[, lapply(.SD, function(x){ifelse(is.na(x), 99, x)})]

fit <- lm(Y~cont1+cont2+cont3+cont4+cont5+cont6+cont7+cont8+cont9+cont10+cont11+cont12+cont13+cont14, data=train)
summary(fit)
#qqPlot(fit)

features_to_drop <- c("cont3", "cont5", "cont6")
all_data <- all_data[, -features_to_drop, with = FALSE]
all_data.sparse <- sparseMatrix(all_data$i, all_data[, cat_features[1], with = FALSE][[1]])
for(var in cat_features[-1]){
  all_data.sparse <- cbind(all_data.sparse, sparseMatrix(all_data$i, all_data[, var, with = FALSE][[1]])) 
}
all_data.sparse <- cbind(all_data.sparse, as.matrix(all_data[,-c(cat_features, 'i'), with = FALSE]))
dim(all_data.sparse)

fairobj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  c <- 2 
  x <-  preds-labels
  grad <- c*x / (abs(x)+c)
  hess <- c^2 / (abs(x)+c)^2
  return(list(grad = grad, hess = hess))
}

xg_eval_mae <- function (yhat, dtrain) {
  y <- xgboost::getinfo(dtrain, "label")
  err <- mae(exp(y),exp(yhat) )
  return (list(metric = "error", value = err))
}

x_train <- all_data.sparse[1:row.train,]
x_test <- all_data.sparse[(row.train+1):nrow(all_data),]

sample.index <- sample(1:nrow(train), nrow(train) * 0.2)
dvalid <- xgb.DMatrix(x_train[-sample.index,], label = Y[-sample.index])
dtrain <- xgb.DMatrix(x_train, label = Y)
dtest <- xgb.DMatrix(x_test)
rm(train, test, x_train, x_test, all_data)

set.seed(7579)
xgb_params <- list(
  colsample_bytree = 0.5, #0.5
  subsample = 0.8, #0.8
  eta = 0.01, #0.01
  objective = fairobj, #'reg:linear'
  max_depth = 12,
  alpha = 1,
  gamma = 2,
  min_child_weight = 1, #1
  base_score = 7.76
)

#set.seed(7579)
#m1 <- xgb.cv(xgb_params,
#             dtrain,
#             nrounds = 8000, 
#             nfold = 5,
#             early_stopping_rounds = 15,
#             print_every_n = 10,
#             verbose = 1,
#             feval = xg_eval_mae,
#             maximize = FALSE)

#best_nrounds <- which.min(m1[, test.error.mean]) .4
#best_nrounds <- m1$best_iteration  .6
#cv_mean <- m1$evaluation_log$test_error_mean[best_nrounds]
#cv_std <- m1$evaluation_log$test_error_std[best_nrounds]
#cat(paste0('CV-Mean: ',cv_mean,' ', cv_std))

set.seed(7579)
gbdt <- xgb.train(xgb_params, 
                  dtrain, 
                  nrounds = 20000,
                  watchlist = list(eval = dvalid),
                  verbose = 1,
                  feval = xg_eval_mae,
                  maximize = F)

sub <- fread("/Users/usagi/Desktop/kaggle/allstate/sample_submission.csv", colClasses = c("integer", "numeric"))
sub$loss <- exp(predict(gbdt,dtest)) - 200
write.csv(sub,'/Users/usagi/Desktop/kaggle/allstate/sub_allsate.csv', row.names = FALSE)

#mae score
#valid_score-error:1040.29399896289  LB: 1116.31221   nrounds = 3813
#valid_score-error:1028.60652821577  LB: 1114.70086   nrounds = 6000
#valid_score-error:993.534451222869  LB: 1112.85508   nrounds = 6000
#valid_score-error:1033.2612083168   LB: 1111.82405   nrounds = 8000
#valid_score-error:1031.54716974128   LB: 1111.27783   nrounds = 10000
#valid_score-error:1021.39748297272   LB: 1110.52161   nrounds = 15000