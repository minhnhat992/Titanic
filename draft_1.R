library(caret)
library(MASS)
library(plyr)
library(dplyr)
library(utils)
library(data.table)
library(doParallel)
library(pROC)
library(Amelia)
library(stringr)

#load file 
train <- fread("F:/R/Titanic/train.csv")

#class
class(train)

#check missing data
sum(is.na(train))

#check which collumn has missing data
missing <- sapply(train,function(x)sum(is.na(x))) %>% 
  data.table()

#get title from name collumn http://www.txt2re.com/index-python.php3?s=%2C+Mr.&submit=Show+Matches
# get character between , and .
train[,"Name"] <- sapply(train[,"Name", with = FALSE],  FUN =function(x) stringr::str_match(x, paste("\\,.*?\\.",sep=""))[,-2]) %>% 
  data.table()
#remove  ,
train[,"Name"] <- sapply(train[,"Name", with = FALSE],  FUN =function(x) stringr::str_replace(x, "[\\W]" , " ")) %>% 
  data.table()


preProcess<- preProcess(x = train[,c("Age", "Fare"), with = FALSE], 
                        method = "knnImpute",
                        na.remove = TRUE,
                        k = 5,
                        knnSummary = mean,
                        outcome = NULL,
                        fudge = .2,
                        numUnique = 3)

train[,c("Age","Fare")] <- predict(preProcess, 
                                   train[,c("Age","Fare"), 
                                         with = FALSE])



#remove id table , ticket and cabin and Name
train[,c("PassengerId" ,"Ticket" ,"Cabin", "Name"):= NULL]

#set collum types
train$Survived <- factor(ifelse(train$Survived == 1, "y", "n"))
train$Pclass <- ordered(train$Pclass, levels = 1:3)
train$Sex <- factor(ifelse(train$Sex == "male", "1", "0"))
sapply(train[,.(SibSp,Age,Parch,Fare)], function(x) as.numeric())
train$Embarked <- factor(train$Embarked, ordered = FALSE)

#deal with NA with amelia
col <- colnames(train)
a.out <- amelia(x = train,
                m = 5,
                p2s = 1,
                noms = c("Survived","Sex","Embarked"),
                ords = "Pclass",
                parallel = "snow",
                empri = 0.5*nrow(train))

#new data
new_train <- data.table(a.out$imputations[[1]])

#split trainning set 70:30 ratio
set.seed(1234)
splitindex <- createDataPartition(train$Survived, 
                                  p = 0.7, 
                                  list = FALSE)
train_set <- train[splitindex,]
valid_set <- train[-splitindex,]

#data set is balanced
print(prop.table(table(train$Survived))*100)

#set up train control
control <- trainControl(method = "cv",
                        repeats = 3,
                        number = 10,
                        verboseIter = TRUE,
                        savePredictions = TRUE,
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE,
                        allowParallel = TRUE)


#create model
#set time
ptm <- proc.time()
set.seed(1234)

#parrallel
cl <- makeCluster(detectCores())
registerDoParallel(cl)

#establish model
model <- caret::train(data = train_set,
                      Survived~.,
                      method = "rf",
                      #preProcess = c("center","scale","pca"),
                      na.action = na.omit,
                      metric = "ROC",
                      trControl = control,
                      verbose = TRUE,
                      size = 3,
                      tuneLength = 5)

#summary model
summary(model)

#stop Cluster
stopCluster(cl)

#stop recording time
time <- proc.time() - ptm

#train set performance
probs <- predict(model,newdata = train_set, type ="prob")
pred  <- factor(ifelse(probs[,"y"] > 0.5,"y","n"))
summary(pred)
matrix <- confusionMatrix(data = pred, 
                           train_set$Survived,
                           positive = levels(train_set$Survived)[2]) %>% 
  print()
rocCurve  <- roc(response = train_set$Survived,
                  predictor = probs[,"y"],
                  levels = levels(train_set$Survived))
curve <- plot(rocCurve, print.thres = c(.5), type = "S",
                print.thres.pattern = "%.3f (Spec = %.2f, Sens = %.2f)",
                print.thres.cex = .8,
                legacy.axes = TRUE)
curve#AUC


#valid_set performance
probs_1 <- predict(model,newdata = valid_set, type ="prob")
pred_1  <- factor(ifelse(probs_1[,"y"] > 0.5,"y","n"))
summary(pred_1)
levels(valid_set$Survived)
matrix_1 <- confusionMatrix(data = pred_1, 
                            valid_set$Survived,
                            positive = levels(valid_set$Survived)[2]) %>% 
  print()

rocCurve_1  <- roc(response = valid_set$Survived,
                   predictor = probs_1[,"y"],
                   levels = levels(valid_set$Survived))
curve_1 <- plot(rocCurve_1, print.thres = c(.25), type = "S",
               print.thres.pattern = "%.3f (Spec = %.2f, Sens = %.2f)",
               print.thres.cex = .8,
               legacy.axes = TRUE)
curve_1

#test set
test <- fread("F:/R/Titanic/test.csv")
pasid <- test[,"PassengerId", with = FALSE]

#remove id table , ticket and cabin and Name
test[,c("PassengerId" ,"Ticket" ,"Cabin", "Name"):= NULL]

#set collum types
test$Pclass <- ordered(test$Pclass, levels = 1:3)
test$Sex <- factor(ifelse(test$Sex == "male", "1", "0"))
sapply(test[,.(SibSp,Age,Parch,Fare)], function(x) as.numeric())
test$Embarked <- factor(test$Embarked, ordered = FALSE)

#missing values
preProcess<- preProcess(x = test[,c("Age", "Fare"), with = FALSE], 
                        method = "knnImpute",
                        na.remove = TRUE,
                        k = 5,
                        knnSummary = mean,
                        outcome = NULL,
                        fudge = .2,
                        numUnique = 3)
test[,c("Age","Fare")] <- predict(preProcess, test[,c("Age","Fare"), with = FALSE])

#predict
test_pred <- predict(model,
                     newdata = test,
                     type = "prob")

prediction <- factor(ifelse(test_pred[,"y"]>0.5,"1","0")) %>% 
  data.table()

final <- cbind(pasid,prediction)  
colnames(final) <- c("PassengerID","Survived")

#final file
write.csv(final,"F:/R/Titanic/gendermodel.csv")

