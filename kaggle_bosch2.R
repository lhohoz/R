library(data.table)
library(xgboost)
library(Matrix)
library(caret)

##Top categorical and numeric features as indicated by tree models
category_features <- c("L3_S32_F3854", "L3_S32_F3851", "L1_S24_F1523", "L1_S24_F1525", "L2_S26_F3038", "L1_S24_F1530",
                       "L1_S24_F819",  "L2_S27_F3131", "L3_S29_F3475", "L3_S29_F3317", "L3_S32_F3853")

date_features <- c("L3_S30_D3496", "L3_S30_D3506", "L3_S30_D3501", 
                   "L3_S32_D3852", "L3_S33_D3856", "L3_S34_D3877")

numeric_features <- c('L1_S24_F1846', 'L3_S32_F3850', 'L1_S24_F1695', 'L1_S24_F1632',
                      'L3_S33_F3855', 'L1_S24_F1604', 'L3_S29_F3407', 'L3_S33_F3865',
                      'L3_S38_F3952', 'L1_S24_F1723')
##Load in Data
dt_cat <- fread("/Users/usagi/Desktop/kaggle/bosch/train_categorical.csv", select = c(category_features))

dt_date <- fread("/Users/usagi/Desktop/kaggle/bosch/train_date.csv", select = c(date_features))

dt_num <- fread("/Users/usagi/Desktop/kaggle/bosch/train_numeric.csv", select = c(numeric_features, "Response"))

dt_cat_test <- fread("/Users/usagi/Desktop/kaggle/bosch/test_categorical.csv", select = c(category_features))

dt_date_test <- fread("/Users/usagi/Desktop/kaggle/bosch/test_date.csv", select = c(date_features))

dt_num_test <- fread("/Users/usagi/Desktop/kaggle/bosch/test_numeric.csv", select = c(numeric_features, "Id"))

Y <- dt_num$Response
dt_num[ , Response := NULL]
Id  <- dt_num_test$Id
dt_num_test[ , Id := NULL]
row.train <- nrow(dt_num)

D.cat <- rbind(dt_cat, dt_cat_test)
D.date <- rbind(dt_date, dt_date_test)
D.num <- rbind(dt_num, dt_num_test)
rm(dt_cat, dt_cat_test, dt_num, dt_num_test, dt_date, dt_date_test)

##Convert columns set NA to 0 and add +2 offset to values
for(col in names(D.num)) set(D.num, j = col, value = D.num[[col]] + 2)
for(col in names(D.num)) set(D.num, which(is.na(D.num[[col]])), col, 9999999) #0
for(col in names(D.date)) set(D.date, which(is.na(D.date[[col]])), col, mean(D.date[[col]], na.rm=T))

#convert categories to numeric integers
for (f in category_features){
  if (class(D.cat[[f]])=='character'){
    levels <- unique(c(D.cat[[f]]))
    D.cat[[f]]<-as.numeric(factor(D.cat[[f]],levels = levels))
  }
}

#Seperate test and training
train.cat <- D.cat[1:row.train,]
test.cat <- D.cat[(row.train+1):nrow(D.cat),]
rm(D.cat)

train.num <- D.num[1:row.train,]
test.num <- D.num[(row.train+1):nrow(D.num),]
rm(D.num)

train.date <- D.date[1:row.train, ]
test.date <- D.date[(row.train+1):nrow(D.date),]
rm(D.date)

#combine categorical and numeric features for test and train
train_sub2 <- cbind(train.num, train.cat, train.date, train[, .(StartTime, EndTime, Timespan, numberOfSteps, longestStep, magic, feature1, feature2, feature3, feature4)])
rm(train.num, train.cat, train.date)

test_sub2 <- cbind(test.num, test.cat, test.date, test[, .(StartTime, EndTime, Timespan, numberOfSteps, longestStep, magic, feature1, feature2, feature3, feature4)])
rm(test.num, test.cat, test.date)

cat("train_col_num:", length(names(train_sub2)))

X.train <- Matrix(as.matrix(train_sub2), sparse = T)
X.test <- Matrix(as.matrix(test_sub2), sparse = T)
rm(test_sub2,train_sub2)

##create folds or CV
set.seed(7579)
folds <- createFolds(as.factor(Y), k = 6)
valid <- folds$Fold3
model <- c(1:length(Y))[-valid]
prior <- sum(Y) / (1* length(Y))

#MCC Score
mcc <- function(y_true, y_prob) {
  DT <- data.table(y_true = y_true, y_prob = y_prob, key="y_prob")
  
  nump <- sum(y_true)
  numn <- length(y_true)- nump
  
  DT[, tn_v:= cumsum(as.numeric(y_true == 0))]
  DT[, fp_v:= cumsum(as.numeric(y_true == 1))]
  DT[, fn_v:= numn - tn_v]
  DT[, tp_v:= nump - fp_v]
  DT[, tp_v:= nump - fp_v]
  DT[, mcc_v:= (tp_v * tn_v - fp_v * fn_v) / sqrt((tp_v + fp_v) * (tp_v + fn_v) * (tn_v + fp_v) * (tn_v + fn_v))]
  DT[, mcc_v:= ifelse(!is.finite(mcc_v), 0, mcc_v)]
  
  return(max(DT[['mcc_v']]))
}
mcc_eval <- function(y_prob, dtrain) {
  y_true <- getinfo(dtrain, "label")
  best_mcc <- mcc(y_true, y_prob)
  return(list(metric="MCC", value=best_mcc))
}

##Parameters
param <- list(objective = "binary:logistic",
              silent = 1,
              colsample_bytree = 0.8, #0.7, 0.8 LB score up
              subsample = 0.7, #0.7
              learning_rate = 0.1, #0.1
              eval_metric = "auc",
              num_parallel_tree = 1,
              max_depth = 4, #4
              min_child_weight = 2, #2
              base_score = prior)

dmodel <- xgb.DMatrix(X.train, label = Y)
#dmodel <- xgb.DMatrix(X.train[model,], label = Y[model])
dvalid <- xgb.DMatrix(X.train[valid,], label = Y[valid])

##Train Model
set.seed(7579)
m1 <- xgb.train(data=dmodel,
                param,
                watchlist = list(mod = dmodel, val = dvalid), 
                nfold = 4, 
                nrounds = 460, #634
                #feval = mcc_eval,
                early_stopping_rounds = 1)

####Hypertuning XGBoost parameters#######
#searchGridSubCol <- expand.grid(subsample = c(0.5, 0.8, 1), 
#                                colsample_bytree = c(0.6, 0.8, 1))

#rmseErrorsHyperparameters <- apply(searchGridSubCol, 1, function(parameterList){
#  currentSubsampleRate <- parameterList[["subsample"]]
#  currentColsampleRate <- parameterList[["colsample_bytree"]]
  
#  xgboostModelCV <- xgb.cv(data=dmodel,
#                           metrics = "rmse", 
#                           verbose = TRUE, 
#                           eval_metric = "rmse",
#                           objective = "binary:logistic",
#                           max_depth = 4, 
#                           min_child_weight = 2,
#                           num_parallel_tree = 1,
#                           learning_rate = 0.05, 
#                           "subsample" = currentSubsampleRate,
#                           "colsample_bytree" = currentColsampleRate,
#                           watchlist = list(mod = dmodel, val = dvalid), 
#                           nfold = 4, 
#                           nrounds = 25,
#                           early_stopping_rounds = 1)
  
#  xvalidationScores <- as.data.frame(xgboostModelCV)
#  rmse <- tail(xvalidationScores$test.rmse.mean, 1)
#  return(c(rmse, currentSubsampleRate, currentColsampleRate))
#})

#xgb.cv
#which(m2$test.MCC.mean == max(m2$test.MCC.mean))

#features importance
imp <- xgb.importance(model = m1, feature_names = colnames(X.train))

pred <- predict(m1, dvalid)
rm(X.train)
rm(dmodel, dvalid)

##Matthews Coefficient to determine threshold
mc <- function(actual, predicted) {
  tp <- as.numeric(sum(actual == 1 & predicted == 1))
  tn <- as.numeric(sum(actual == 0 & predicted == 0))
  fp <- as.numeric(sum(actual == 0 & predicted == 1))
  fn <- as.numeric(sum(actual == 1 & predicted == 0))
  numer <- (tp * tn) - (fp * fn)
  denom <- ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ^ 0.5
  numer / denom
}
matt <- data.table(thresh = seq(0.990, 0.999, by = 0.0001))
matt$scores <- sapply(matt$thresh, function(x) mc(Y[valid], (pred > quantile(pred, x)) * 1))
print(matt)
best <- matt$thresh[which(matt$scores == max(matt$scores))]
print(matt$scores[which(matt$scores == max(matt$scores))])

##Create Submission
dtest <- xgb.DMatrix(X.test)
pred  <- predict(m1, dtest)
summary(pred)
sub   <- data.table(Id = Id,
                    Response = (pred > quantile(pred, best)) * 1)
write.csv(sub, "/Users/usagi/Desktop/kaggle/bosch/sub_bosch2.csv", row.names = F)

rm(list=setdiff(ls(), c("train", "test", "imp", "feature_summary", "m2")))


#[633]	mod-auc:0.946298	val-auc:0.944532 CV:0.4576599  LB:0.41476  FE:34 nround = 634
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
#0.0000022 0.0006206 0.0010710 0.0057930 0.0019480 0.9999000 

#[579]	mod-auc:0.944563	val-auc:0.942659  CV:0.4558643  LB:0.41329  FE:34 nround = 580
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
#0.0000026 0.0006305 0.0010750 0.0057890 0.0019440 0.9999000 

#[429]	mod-auc:0.938347	val-auc:0.937018 CV:0.4404136  LB:0.40982  FE:34 nround = 430
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
#0.0000032 0.0006688 0.0011150 0.0058070 0.0019430 0.9999000 

#[299]	mod-auc:0.931581	val-auc:0.929338 CV:0.4304104  LB:0.40452  FE:34 nround = 300
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
#0.0000035 0.0007166 0.0011570 0.0058120 0.0019800 0.9998000 

#[231]	mod-auc:0.927761	val-auc:0.925464 CV:0.420776  LB:0.40265  FE:34 nround = 232
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
#0.0000041 0.0007349 0.0011850 0.0057980 0.0019920 0.9998000 