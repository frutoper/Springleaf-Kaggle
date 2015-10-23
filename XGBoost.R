setwd("D://OIE2_mfruin//Kaggle//Springleaf")

library(xgboost)
library(FeatureHashing)
library(gbm)

#train <- read.csv("Data\\train.csv")

train <- readRDS("Data//Train.rds")

Target <- train$target

### ---------- Create Validation Set ----------------------------------------------------------------###

tr <- train[1:(dim(train)[1]/2),]
val <- train[((dim(train)[1]/2)+1):(dim(train)[1]),]

trTarget <- tr$target
valTarget <- val$target
# ----------------- Create Sparce Matrices -------------------------------------------------------------

x <- names(train[,!(names(train) %in% c("target"))])
Formula <- as.formula(paste("target", paste(x, collapse='+'), sep='~'))
f <- paste(x, collapse=' + ')
f2 <- paste('~', f, sep = ' ')

m.train <- hashed.model.matrix(f2, train, 2^16)
rm(train)


m.tr <- hashed.model.matrix(f2, tr, 2^16)
m.val <- hashed.model.matrix(f2, val, 2^16)
rm(tr)
rm(val)

##  ------------------- Create Random HyperParameters -------------------------------------------------
set.seed(1)
iter <- 75
RandomSearch <- data.frame(max.depth = runif(iter, 7, 13), 
                           nround = runif(iter, 500, 1200),
                           eta = runif(iter, .00001, .06),
                           booster = c("gbtree"),
                           min_child_weight = runif(iter, .2, 1.5),
                           subsample = runif(iter, .85, 1),
                           colsample_bytree = runif(iter, 0.001, 1),
                           lambda = runif(iter, .01, 3),
                           alpha  = runif(iter, 1, 3.5)                         
                           )
RandomSearch$AUC <- 0

##  ------------------- Iterate Through HyperParameters -------------------------------------------------

for(i in 1:iter){
  print(i)
  # XGBoost
  FitXGB <- xgboost(m.tr, trTarget, 
                    max.depth=RandomSearch[i,1], 
                    nround = RandomSearch[i,2], 
                    eta=RandomSearch[i,3],
                    objective = "binary:logistic", 
                    verbose = ifelse(interactive(), 1, 0),
                    booster = as.character(RandomSearch[i, 4]),
                    min_child_weight = RandomSearch[i,5],
                    subsample = RandomSearch[i, 6],
                    colsample_bytree = RandomSearch[i, 7],
                    lambda = RandomSearch[i, 8],
                    alpha = RandomSearch[i, 9],
                    eval_metric = "auc")
  
  #------------------------Score/Validate Model----------------------------------#
  
  Predxgb <- predict(FitXGB, m.val)
  auc <- gbm.roc.area(valTarget, Predxgb)
  RandomSearch$AUC[i] <- auc
}

rm(m.tr)
rm(m.val)
rm(FitXGB)
# ------------------ Save Hyper Parameters ----------------------------------------------------- #

write.csv(RandomSearch, paste0("Output//Tune XGBOOST ", iter, " target_more_hyper.csv"), row.names = F)

BestRow <- which.max(RandomSearch[,"AUC"] )

# ---------------------- Read in Test Data, Create Hashed Matrix -----------------------------#

test <- read.csv("Data\\test.csv")

m.test <- hashed.model.matrix(f2, test, 2^16)
IDs <- test$ID

rm(test)
# ---------------- Final model with best HyperParameters -----------------------------#

BestXGB <- xgboost(m.train, Target, 
                  max.depth=RandomSearch[BestRow,1],
                  nround = RandomSearch[BestRow,2], 
                  eta=RandomSearch[BestRow,3],
                  objective = "binary:logistic", 
                  verbose = ifelse(interactive(), 1, 0),
                  booster = as.character(RandomSearch[BestRow, 4]),
                  min_child_weight = RandomSearch[BestRow,5],
                  subsample = RandomSearch[BestRow, 6],
                  colsample_bytree = RandomSearch[BestRow, 7],
                  lambda = RandomSearch[BestRow, 8],
                  alpha = RandomSearch[BestRow, 9],
                  eval_metric = "auc")


target <- predict(BestXGB, m.test)
sub <- data.frame(ID = IDs, target = target)

write.csv(sub, paste0("Data\\Submit_XGB_", iter, "_more_hyper.csv") , row.names = F)

q(n)