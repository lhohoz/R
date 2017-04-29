library(jsonlite)#read json
library(dplyr)
library(caret)
library(reshape2)
library(lubridate)
library(purrr)
library(tidytext)
library(reshape2)
library(xgboost)
library(data.table)
library(stringr)
#library(syuzhet) #text mining

catNWayAvgCV <- function(data, varList, y, pred0, filter, k, f, g=42, lambda=NULL, r_k, cv=NULL){ #g=42
  # It is probably best to sort your dataset first by filter and then by ID (or index)
  n <- length(varList)
  varNames <- paste0("v",seq(n))
  ind <- unlist(cv, use.names=FALSE)
  oof <- NULL
  if (length(cv) > 0){
    for (i in 1:length(cv)){
      sub1 <- data.table(v1=data[,varList,with=FALSE], y=data[,y,with=FALSE], pred0=data[,pred0,with=FALSE], filt=filter)
      sub1 <- sub1[sub1$filt==TRUE,]
      sub1[,filt:=NULL]
      colnames(sub1) <- c(varNames,"y","pred0")
      sub2 <- sub1[cv[[i]],]
      sub1 <- sub1[-cv[[i]],]
      sum1 <- sub1[,list(sumy=sum(y), avgY=mean(y), cnt=length(y)), by=varNames]
      tmp1 <- merge(sub2, sum1, by = varNames, all.x=TRUE, sort=FALSE)
      set(tmp1, i=which(is.na(tmp1[,cnt])), j="cnt", value=0)
      set(tmp1, i=which(is.na(tmp1[,sumy])), j="sumy", value=0)
      if(!is.null(lambda)) tmp1[beta:=lambda] else tmp1[,beta:= 1/(g+exp((tmp1[,cnt] - k)/f))]
      tmp1[,adj_avg:=((1-beta)*avgY+beta*pred0)]
      set(tmp1, i=which(is.na(tmp1[["avgY"]])), j="avgY", value=tmp1[is.na(tmp1[["avgY"]]), pred0])
      set(tmp1, i=which(is.na(tmp1[["adj_avg"]])), j="adj_avg", value=tmp1[is.na(tmp1[["adj_avg"]]), pred0])
      set(tmp1, i=NULL, j="adj_avg", value=tmp1$adj_avg*(1+(runif(nrow(sub2))-0.6)*r_k)) #0.6
      oof <- c(oof, tmp1$adj_avg)
    }
  }
  oofInd <- data.frame(ind, oof)
  oofInd <- oofInd[order(oofInd$ind),]
  sub1 <- data.table(v1=data[,varList,with=FALSE], y=data[,y,with=FALSE], pred0=data[,pred0,with=FALSE], filt=filter)
  colnames(sub1) <- c(varNames,"y","pred0","filt")
  sub2 <- sub1[sub1$filt==F,]
  sub1 <- sub1[sub1$filt==T,]
  sum1 <- sub1[,list(sumy=sum(y), avgY=mean(y), cnt=length(y)), by=varNames]
  tmp1 <- merge(sub2, sum1, by = varNames, all.x=TRUE, sort=FALSE)
  tmp1$cnt[is.na(tmp1$cnt)] <- 0
  tmp1$sumy[is.na(tmp1$sumy)] <- 0
  if(!is.null(lambda)) tmp1$beta <- lambda else tmp1$beta <- 1/(g+exp((tmp1$cnt - k)/f))
  tmp1$adj_avg <- (1-tmp1$beta)*tmp1$avgY + tmp1$beta*tmp1$pred0
  tmp1$avgY[is.na(tmp1$avgY)] <- tmp1$pred0[is.na(tmp1$avgY)]
  tmp1$adj_avg[is.na(tmp1$adj_avg)] <- tmp1$pred0[is.na(tmp1$adj_avg)]
  # Combine train and test into one vector
  return(c(oofInd$oof, tmp1$adj_avg))
}

address_map <- function(x){
  add <- c("w" = "west", "st."= "street", "ave" = "avenue", "st" = "street",
           "e" = "east", "n" = "north", "s" = "south")
  words_list <- strsplit(x, "\\s+")
  words_list <- unlist(lapply(unlist(words_list), function(x){ifelse(x %in% names(add), x <- add[x], x)}))
  words_list <- paste(words_list, collapse = " ")
  return(words_list)
}

train <- fromJSON("../usagi/Desktop/kaggle/rental/train.json")
test <- fromJSON("../usagi/Desktop/kaggle/rental/test.json")

vars <- setdiff(names(train), c("photos", "features"))
train <- map_at(train, vars, unlist) %>% tibble::as_tibble(.)
train_id <-train$listing_id

vars <- setdiff(names(test), c("photos", "features"))
test <- map_at(test, vars, unlist) %>% tibble::as_tibble(.)
test_id <-test$listing_id

train$filter <- 0
test$filter <- 2

train$feature_count <- lengths(train$features)
test$feature_count <- lengths(test$features)
train$photo_count <- lengths(train$photos)
test$photo_count <- lengths(test$photos)

train[unlist(map(train$features,is_empty)),]$features <- 'nofeat'
test[unlist(map(test$features,is_empty)),]$features <- 'nofeat'
test$interest_level <- 'none'

train_test <- rbind(train,test)

#feats <- data.table(listing_id=rep(unlist(train_test$listing_id), lapply(train_test$features, length)), features=unlist(train_test$features))
#feats[,features:=gsub(" ", "_", paste0("feature_",trimws(char_tolower(features))))]
#feats_summ <- feats[,.N, by=features]
#feats_cast <- dcast.data.table(feats[!features %in% feats_summ[N<10, features]], listing_id ~ features, fun.aggregate = function(x) as.integer(length(x) > 0), value.var = "features")
#train_test <- merge(train_test, feats_cast, by="listing_id", all.x=TRUE, sort=FALSE)

#sentiment <- get_nrc_sentiment(train_test$description)
#sentiment$id <- seq(1:nrow(sentiment))
#train_test$id <- seq(1:length(train_test$building_id))
#train_test <- merge(train_test, sentiment, by.x="id", by.y="id", all.x=T, all.y=T)
#rm(sentiment);gc()

train_test$pred0_low <- sum(train_test$interest_level=="low")/sum(train_test$filter==0)
train_test$pred0_medium <- sum(train_test$interest_level=="medium")/sum(train_test$filter==0)
train_test$pred0_high <- sum(train_test$interest_level=="high")/sum(train_test$filter==0)

train_test$dummy <- "A"
train_test$low <- as.integer(train_test$interest_level=="low")
train_test$medium <- as.integer(train_test$interest_level=="medium")
train_test$high <- as.integer(train_test$interest_level=="high")

set.seed(321)
cvFoldsList <- createFolds(train$interest_level, k=5, list=TRUE, returnTrain=FALSE) #5
train_test <- data.table(train_test)

highCard <- c(
  "building_id",
  "manager_id"
)
for (col in 1:length(highCard)){
  train_test[,paste0(highCard[col],"_mean_med"):=catNWayAvgCV(train_test, varList=c("dummy", highCard[col]), y="medium", pred0="pred0_medium", filter=train_test$filter==0, k=8, f=1, r_k=0.03, cv=cvFoldsList)] #8|1|0.03
  train_test[,paste0(highCard[col],"_mean_high"):=catNWayAvgCV(train_test, varList=c("dummy", highCard[col]), y="high", pred0="pred0_high", filter=train_test$filter==0, k=8, f=1, r_k=0.03, cv=cvFoldsList)] #8|1|0.03
}
train_test <- as.data.frame(train_test)

train_test$add <- sapply(train_test$description, tolower)
train_test$add <- sapply(train_test$add, address_map)
train_test$website_redacted <- sapply(train_test$add, function(x){ifelse(grepl('website_redacted', x), 1, 0)})
train_test$street <- sapply(train_test$add, function(x){ifelse(grepl('street', x), 1, 0)})
train_test$avenue <- sapply(train_test$add, function(x){ifelse(grepl('avenue', x), 1, 0)})
train_test$east <- sapply(train_test$add, function(x){ifelse(grepl('east', x), 1, 0)})
train_test$west <- sapply(train_test$add, function(x){ifelse(grepl('west', x), 1, 0)})
train_test$north <- sapply(train_test$add, function(x){ifelse(grepl('north', x), 1, 0)})
train_test$south <- sapply(train_test$add, function(x){ifelse(grepl('south', x), 1, 0)})
train_test$other_address <- ifelse(apply(train_test[c('street', 'avenue', 'east', 'west', 'north', 'south')], 1, sum) == 0, 1, 0)

feat <- c("bathrooms", "bedrooms", "building_id", "created", "latitude", "description",
          "listing_id", "longitude", "manager_id", "price", "features",
          "display_address", "street_address", "feature_count", "photo_count", "interest_level", 
          'add', 'street', 'avenue', 'east', 'west', 'north', 'south', 'website_redacted',
          'other_address', 'building_id_mean_med', 'building_id_mean_high', 'manager_id_mean_med', 'manager_id_mean_high')

train_test <- train_test[, names(train_test) %in% feat]

word_remove <- c('allowed', 'building', 'center', 'space', '2', '2br', 'bldg', '24',
                '3br', '1', 'ft', '3', '7', '1br', 'hour', 'bedrooms', 'true', 'bedroom',
                'stop', 'size', 'blk', '4br', '4', 'sq', '0862', '1.5', '373', '16', '3rd', 'block',
                'st', '01', 'bathrooms', 'price')

word_sparse <- train_test[,names(train_test) %in% c("features", "listing_id")]
train_test$features <- NULL

word_sparse <- word_sparse %>%
  filter(map(features, is_empty) != TRUE) %>%
  tidyr::unnest(features) %>%
  unnest_tokens(word, features)

data("stop_words")

word_sparse <- word_sparse[!(word_sparse$word %in% stop_words$word),]
word_sparse <- word_sparse[!(word_sparse$word %in% word_remove),]

top_word <- as.character(as.data.frame(sort(table(word_sparse$word), decreasing=TRUE)[1:500])$Var1) #500
word_sparse <- word_sparse[word_sparse$word %in% top_word,]
word_sparse$word <- as.factor(word_sparse$word)
word_sparse <- dcast(word_sparse, listing_id ~ word, length, value.var = "word")
train_test <- merge(train_test, word_sparse, by = "listing_id", sort = FALSE, all.x=TRUE)

time <- fread("../usagi/Desktop/kaggle/rental/image_time.csv")
time[, img_date:=NULL]
time[, time_stamp:=NULL]
train_test <- merge(train_test, time, by="listing_id", sort=FALSE, all.x=TRUE)

train_test$building_id <- as.integer(factor(train_test$building_id))
train_test$manager_id <- as.integer(factor(train_test$manager_id))
train_test$display_address <- as.integer(factor(train_test$display_address))
train_test$street_address <- as.integer(factor(train_test$street_address))

train_test$created <- ymd_hms(train_test$created)
train_test$month <- month(train_test$created)
train_test$day <- day(train_test$created)
train_test$hour <- hour(train_test$created)
train_test$created <- NULL

train_test$description_len <- sapply(strsplit(train_test$description, "\\s+"), length)
train_test$description <- NULL
train_test$add <- NULL

train_test$bed_price <- train_test$price / train_test$bedrooms
train_test[which(is.infinite(train_test$bed_price)),]$bed_price <- train_test[which(is.infinite(train_test$bed_price)),]$price
train_test$bath_price <- train_test$price / train_test$bathrooms
train_test[which(is.infinite(train_test$bath_price)),]$bath_price <- train_test[which(is.infinite(train_test$bath_price)),]$price

train_test$room_sum <- train_test$bedrooms + train_test$bathrooms
train_test$room_diff <- train_test$bedrooms - train_test$bathrooms
train_test$room_price <- train_test$price / train_test$room_sum
train_test$bed_ratio <- train_test$bedrooms / train_test$room_sum
train_test[which(is.infinite(train_test$room_price)),]$room_price <- train_test[which(is.infinite(train_test$room_price)),]$price

train_test$photo_count <- log1p(train_test$photo_count)
train_test$feature_count <- log1p(train_test$feature_count)
train_test$price <- log1p(train_test$price)
train_test$room_price <- log1p(train_test$room_price)
train_test$bed_price <- log1p(train_test$bed_price)
train_test$bath_price <- log1p(train_test$bath_price)

train_test <- data.table(train_test)
manager_per <- train_test[, .(count = .N), by=manager_id]
building_per <- train_test[, .(count = .N), by=building_id]

mid_10 <- manager_per[count >= quantile(manager_per$count, c(.90)), manager_id] 
mid_25 <- manager_per[count >= quantile(manager_per$count, c(.75)), manager_id]
mid_5 <- manager_per[count >= quantile(manager_per$count, c(.95)), manager_id]
mid_50 <- manager_per[count >= quantile(manager_per$count, c(.50)), manager_id]
mid_1 <- manager_per[count >= quantile(manager_per$count, c(.99)), manager_id]
mid_2 <- manager_per[count >= quantile(manager_per$count, c(.98)), manager_id]
mid_15 <- manager_per[count >= quantile(manager_per$count, c(.85)), manager_id]
mid_20 <- manager_per[count >= quantile(manager_per$count, c(.80)), manager_id]
mid_30 <- manager_per[count >= quantile(manager_per$count, c(.70)), manager_id]

bid_10 <- building_per[count >= quantile(building_per$count, c(.90)), building_id]
bid_25 <- building_per[count >= quantile(building_per$count, c(.75)), building_id]
bid_5 <- building_per[count >= quantile(building_per$count, c(.95)), building_id]
bid_50 <- building_per[count >= quantile(building_per$count, c(.50)), building_id]
bid_1 <- building_per[count >= quantile(building_per$count, c(.99)), building_id]
bid_2 <- building_per[count >= quantile(building_per$count, c(.98)), building_id]
bid_15 <- building_per[count >= quantile(building_per$count, c(.85)), building_id]
bid_20 <- building_per[count >= quantile(building_per$count, c(.80)), building_id]
bid_30 <- building_per[count >= quantile(building_per$count, c(.70)), building_id]

train_test$top_10_manager <- sapply(train_test$manager_id, function(x){ifelse(x %in% mid_10, 1, 0)})
train_test$top_25_manager <- sapply(train_test$manager_id, function(x){ifelse(x %in% mid_25, 1, 0)})
train_test$top_5_manager <- sapply(train_test$manager_id, function(x){ifelse(x %in% mid_5, 1, 0)})
train_test$top_50_manager <- sapply(train_test$manager_id, function(x){ifelse(x %in% mid_50, 1, 0)})
train_test$top_1_manager <- sapply(train_test$manager_id, function(x){ifelse(x %in% mid_1, 1, 0)})
train_test$top_2_manager <- sapply(train_test$manager_id, function(x){ifelse(x %in% mid_2, 1, 0)})
train_test$top_15_manager <- sapply(train_test$manager_id, function(x){ifelse(x %in% mid_15, 1, 0)})
train_test$top_20_manager <- sapply(train_test$manager_id, function(x){ifelse(x %in% mid_20, 1, 0)})
train_test$top_30_manager <- sapply(train_test$manager_id, function(x){ifelse(x %in% mid_30, 1, 0)})

train_test$top_10_building <- sapply(train_test$building_id, function(x){ifelse(x %in% bid_10, 1, 0)})
train_test$top_25_building <- sapply(train_test$building_id, function(x){ifelse(x %in% bid_25, 1, 0)})
train_test$top_5_building <- sapply(train_test$building_id, function(x){ifelse(x %in% bid_5, 1, 0)})
train_test$top_50_building <- sapply(train_test$building_id, function(x){ifelse(x %in% bid_50, 1, 0)})
train_test$top_1_building <- sapply(train_test$building_id, function(x){ifelse(x %in% bid_1, 1, 0)})
train_test$top_2_building <- sapply(train_test$building_id, function(x){ifelse(x %in% bid_2, 1, 0)})
train_test$top_15_building <- sapply(train_test$building_id, function(x){ifelse(x %in% bid_15, 1, 0)})
train_test$top_20_building <- sapply(train_test$building_id, function(x){ifelse(x %in% bid_20, 1, 0)})
train_test$top_30_building <- sapply(train_test$building_id, function(x){ifelse(x %in% bid_30, 1, 0)})

train <- train_test[train_test$listing_id %in% train_id,]
test <- train_test[train_test$listing_id %in% test_id,]

train$interest_level <- as.integer(factor(train$interest_level))
y <- train$interest_level
y <- y - 1
train[, interest_level:= NULL]
test[, interest_level:= NULL]

set.seed(1985)
param <- list(booster="gbtree",
              objective="multi:softprob",
              eval_metric="mlogloss",
              nthread = 13,
              num_class = 3,
              eta = 0.02,
              gamma = 1,
              max_depth = 4,
              min_child_weight = 1,
              subsample = 0.7,
              colsample_bytree = 0.5
)

for(col in names(test)) set(test, which(is.na(test[[col]])), col, 0)
for(col in names(train)) set(train, which(is.na(train[[col]])), col, 0)

#create folds
#kfolds <- 5
#folds <- createFolds(y, k = kfolds, list = TRUE, returnTrain = FALSE)
#fold <- as.numeric(unlist(folds[1]))
#x_train <- train[-fold,]
#x_val <- train[fold,] 
#y_train <- y[-fold]
#y_val <- y[fold]
#dtrain <- xgb.DMatrix(as.matrix(x_train), label=y_train)
#dval <- xgb.DMatrix(as.matrix(x_val), label=y_val)

dtest <- xgb.DMatrix(data.matrix(test))
dtrain <- xgb.DMatrix(data.matrix(train), label = y)

gbdt <- xgb.train(params = param,
                 data = dtrain,
                 nrounds = 20000, #20000
                 watchlist = list(train = dtrain),
                 #watchlist = list(train = dtrain, val=dval),
                 print_every_n = 25,
                 early_stopping_rounds = 50)

allpredictions <- (as.data.frame(matrix(predict(gbdt, dtest), nrow=dim(test), byrow=TRUE)))
allpredictions <- cbind(allpredictions, test$listing_id)
names(allpredictions) <- c("high","low","medium","listing_id")
allpredictions <- allpredictions[,c(1,3,2,4)]

write.csv(allpredictions,"/Users/usagi/Desktop/kaggle/rental/sub_rental.csv",row.names = FALSE)
#imp <- xgb.importance(names(train), model = gbdt)