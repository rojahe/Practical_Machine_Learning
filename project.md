## Human Activity Recognition for Weight Lifting Exercise
Robert Herbert

12/9/2014

Practical Machine Learning

## Introduction
This study explores prediction of an activity performed by participants in a weight lifting exercise (WLE). Using devices such as Jawbone Up, Nike FuelBand, and Fitbit, it is now possible to collect a large amount of data about personal activity relatively inexpensively. Such devices are part of the 'quantified self' movement - a group of enthusiasts who take measurements about their physical performance regularly to improve their health, to find patterns in their behavior, or because they find exercise telemetry useful. 

Data for this project are gathered from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different exercise activities. The outcome variable to be predicted using these data is the type of exercise activity. The original data are available available at http://groupware.les.inf.puc-rio.br/work.jsf?p1=11201. 

## Overview of Methods
Machine learning algorithms were used to predict the class of activity in the WLE dataset.  Version x64 3.03 of the R software package, running under RStudio Version 0.98.1091, was used for the analysis.   Preprocessing was performed to to remove unneeded or unusable variables. The cleaned data was separated into training and test sets, using a 60%/40%. split. The training dataset had 11,776 observations and 153 variables; the comparable testing set had 7,846 observations and 153 variables.

Several different model types (classification tree, random forest, and gradient boosting machine) were tried on the training dataset to examine prediction accuracy, with cross validation used to estimate the range of model error.  A final model was then fit on a reserved test data set of 20 cases to check the final model accuracy.

## Data Preprocessing
After loading the raw data, a preliminary analysis was performed on all original variables. It was quickly observed that there are seven subject-identification related variables not relevant for prediction; these were removed from all data sets. It was also noted that there were many variables with a large percentage of the values missing for all study participants.  

After an initial assessement of the effect of omitting these variables versus imputation of missing values, a decision was made to omit all variables with any missing values.  Though this reduced the total available predictors from 153 to 52, the scope of the remaining predictors was judged sufficient for modeling.

Assessment was also made of potential collinearity among the predictor variables.  Moderately high correlations were found among some of the predictors, but given the types of modeling to be done, these variables were not removed.  A future analysis that examines models with more stringent assumptions regarding collinearity (e.g., multiple linear regression) might omit some of these variables, or combine them using a principle components approach.

R code for the preprocessing and modeling of the data will be included at the end of this presentation.

## Modeling - Random Forest 
The first model tried was a random forest, using 5-fold crossvalidation.   The results were quite good, with a 99% accuracy for the 5 activity classes.  The out-of-bag (OOB) error, an unbiased estimate of the true prediction error, was found to be 0.9%.
```
Call:
 randomForest(x = x, y = y, mtry = param$mtry) 
               Type of random forest: classification
                     Number of trees: 500
No. of variables tried at each split: 27

        OOB estimate of  error rate: 0.9%
Confusion matrix:
     A    B    C    D    E class.error
A 3342    4    0    0    2 0.001792115
B   19 2250    9    1    0 0.012724879
C    0    8 2034   12    0 0.009737098
D    0    1   30 1895    4 0.018134715
E    0    1    6    9 2149 0.007390300

> modRF
Random Forest 

11776 samples
   52 predictors
    5 classes: 'A', 'B', 'C', 'D', 'E' 

## Modeling - Random Forest 

No pre-processing
Resampling: Cross-Validated (5 fold) 

Summary of sample sizes: 9420, 9421, 9420, 9420, 9423 

Resampling results across tuning parameters:

  mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
  2     0.988     0.984  0.00151      0.00191 
  27    0.989     0.986  0.00223      0.00282 
  52    0.986     0.982  0.00305      0.00386 

Accuracy was used to select the optimal model using  the largest value.
The final value used for the model was mtry = 27. 


Check model accuracy on test data
> # check prediction accuracy on test set
> pred <- predict(modRF,test)
> table(pred,test$classe)
    
pred    A    B    C    D    E
   A 2228   11    0    0    0
   B    4 1502    6    0    1
   C    0    5 1359   19    3
   D    0    0    3 1265    5
   E    0    0    0    2 1433
> 
> testRF<-predict(modRF,newdata=test)
> cmRF<-confusionMatrix(testRF, test$classe)
> ## accuracy
> cmRF$overall[1]
 Accuracy 
0.9923528 
```

## Modeling - Classification Tree
The second model tried was a classification tree, using the same training set and variables as above. By default, the caret package in R uses bootstrap resampling for this technique, and that
default was used here. These results were not nearly as good, with only a 50% accuracy on the test set. It is worth noting that many of the same variables were key predictors in the tree solution, but the overall predictive performance was not nearly as good.

```
> modTree <- train(classe ~ .,method="rpart",data=train)
Loading required package: rpart
> modTree$finalModel
n= 11776 

node), split, n, loss, yval, (yprob)
      * denotes terminal node

 1) root 11776 8428 A (0.28 0.19 0.17 0.16 0.18)  
   2) roll_belt< 130.5 10799 7457 A (0.31 0.21 0.19 0.18 0.11)  
     4) pitch_forearm< -33.95 945    8 A (0.99 0.0085 0 0 0) *
     5) pitch_forearm>=-33.95 9854 7449 A (0.24 0.23 0.21 0.2 0.12)  
      10) yaw_belt>=169.5 495   47 A (0.91 0.038 0 0.051 0.0061) *
      11) yaw_belt< 169.5 9359 7107 B (0.21 0.24 0.22 0.2 0.13)  
        22) magnet_dumbbell_z< -93.5 1118  467 A (0.58 0.29 0.046 0.055 0.029) *
        23) magnet_dumbbell_z>=-93.5 8241 6238 C (0.16 0.23 0.24 0.22 0.14)  
          46) pitch_belt< -42.95 500   79 B (0.018 0.84 0.094 0.026 0.02) *
          47) pitch_belt>=-42.95 7741 5785 C (0.17 0.19 0.25 0.24 0.15)  
            94) accel_forearm_x>=-99.5 4688 3417 C (0.21 0.22 0.27 0.12 0.18) *
            95) accel_forearm_x< -99.5 3053 1776 D (0.096 0.16 0.22 0.42 0.1) *
   3) roll_belt>=130.5 977    6 E (0.0061 0 0 0 0.99) *
> modTree
CART 

11776 samples
   52 predictors
    5 classes: 'A', 'B', 'C', 'D', 'E' 

No pre-processing
Resampling: Bootstrapped (25 reps) 

Summary of sample sizes: 11776, 11776, 11776, 11776, 11776, 11776, ... 

Resampling results across tuning parameters:

  cp      Accuracy  Kappa   Accuracy SD  Kappa SD
  0.0391  0.511     0.367   0.0684       0.113   
  0.0395  0.499     0.348   0.0727       0.12    
  0.114   0.329     0.0687  0.0406       0.0622  

Accuracy was used to select the optimal model using  the largest value.
The final value used for the model was cp = 0.0391. 
> 
> # check prediction accuracy on test set
> pred <- predict(modTree,newdata=test)
> table(pred,test$classe)

pred    A    B    C    D    E
   A 1357  229   38   66   15
   B    3  259   28    8   10
   C  693  725  819  389  557
   D  171  305  483  823  200
   E    8    0    0    0  660
> 
testTree<-predict(modTree,newdata=test)
cmTree<-confusionMatrix(testTree, test$classe)
accuracy
cmTree$overall[1]

 Accuracy 
0.4993627 
```

## Modeling - Gradient Boosting Machine (GBM)
The final model utilized a Gradient Boosting Machine (GBM) model, using the same training set and variables as above. 
As with the random forest model, this implementation used a 5-fold crossvalidation, and the results were quite similar, 
with a 96% accuracy when applied to the test set. 

```
> modGBM <- train(classe ~ .,method="gbm",data=train, 
+                 trControl = trainControl(method = "cv", number = 5), verbose=FALSE)
> modGBM
Stochastic Gradient Boosting 

11776 samples
   52 predictors
    5 classes: 'A', 'B', 'C', 'D', 'E' 

No pre-processing
Resampling: Cross-Validated (5 fold) 

Summary of sample sizes: 9420, 9421, 9422, 9420, 9421 

Resampling results across tuning parameters:

  interaction.depth  n.trees  Accuracy  Kappa  Accuracy SD  Kappa SD
  1                  50       0.753     0.687  0.00608      0.00751 
  1                  100      0.818     0.77   0.00961      0.0122  
  1                  150      0.852     0.813  0.00889      0.0114  
  2                  50       0.853     0.814  0.0067       0.00854 
  2                  100      0.903     0.877  0.00842      0.0107  
  2                  150      0.928     0.909  0.00679      0.00864 
  3                  50       0.895     0.867  0.0111       0.0141  
  3                  100      0.941     0.926  0.00695      0.00882 
  3                  150      0.959     0.948  0.00513      0.0065  

Tuning parameter 'shrinkage' was held constant at a value of 0.1
Accuracy was used to select the optimal model using  the largest value.
The final values used for the model were n.trees = 150, interaction.depth = 3 and shrinkage = 0.1. 
> 
> testGBM<-predict(modGBM,newdata=test)
> cmGBM<-confusionMatrix(testGBM, test$classe)
> ##Boosting Model Accuracy
> cmGBM$overall[1]
 Accuracy 
0.9627836 
```
## Summary and Conclusion

Based on the above analyses, a very good predictor model was obtained using the random forest approach. 
5-fold crossvalidation was presented, but other values for the technique did not produce significantly 
different results.  The final model worked very well on the test dataset, and showed high accuracy and very low out of sample error. The random forest model was slightly better than the 
GBM model, and was used to predict the final 20 reserved values for testing for this project.

It is worth noting that only a few variables were especially important in the random forest model, when
plotted using the varImp function (see plot, below). Further analyses could work with fewer predictors and attempt to combine correlated predictors in principle
components to see the effect of simplification and data compression on the overall model accuracy.

![](e:\varImp.png)


## R Code for this writeup
```
## environment
library(caret)
library(ggplot2)
library(gridExtra)
library(Hmisc)
library(AppliedPredictiveModeling)
library(randomForest)
set.seed(12345)

## working directory. 
setwd("E:/Practical Machine Learning")

## read raw data; set blank values to NA; same approach for training and test sets
## remove first 7 variables -> not useful for analysis
trainingDataRaw <- read.csv("pml-training.csv",header=TRUE,na.strings=c("NA",""))
trainingDataRaw <- trainingDataRaw[,-c(1:7)]

testingDataRaw <- read.csv("pml-testing.csv",header=TRUE,na.strings=c("NA",""))
testingDataRaw <- testingDataRaw[,-c(1:7)]

length(names(trainingDataRaw))  # variables
nrow(trainingDataRaw)           # observations 

## look at distribution of dependent variable in training set 
table(trainingDataRaw$classe)

## find variables with no missing values
missvar<-names(trainingDataRaw)[apply(trainingDataRaw,2,function(x) 
   table(is.na(x))[1]==nrow(trainingDataRaw))]   

## keep variables with no NA values
trainingDataClean<-trainingDataRaw[,missvar]
testingDataClean<-testingDataRaw[,missvar[-length(missvar)]]


## look for predictor variables with high collinearity (>=.7)
## only use numeric variables ->  ignore for these analyses
## correlations <- cor(trainingDataClean[c(1:52)])      
## collin <- findCorrelation(correlations, cutoff = .7)
## collin

str(trainingDataClean)
summary(trainingDataClean)

# split cleaned set into training and test sets; 60/40
inTrain <- createDataPartition(y=trainingDataClean$classe, p=0.60, list=FALSE)
train <- trainingDataClean[inTrain,]
test  <- trainingDataClean[-inTrain,]
dim(train)
dim(test)


## try a random forest approach
ctrl = trainControl(method = "cv", number = 5)
modRF<-train(classe~.,data=train,method="rf",trControl=ctrl)
modRF$finalModel
modRF

# check prediction accuracy on test set
pred <- predict(modRF,test)
table(pred,test$classe)

testRF<-predict(modRF,newdata=test)
cmRF<-confusionMatrix(testRF, test$classe)
## accuracy
cmRF$overall[1]

## relative variable importance
plot (varImp (modRF, scale = FALSE), top = 20)


## try a classification tree approach
modTree <- train(classe ~ .,method="rpart",data=train)
modTree$finalModel
modTree

# check prediction accuracy on test set
pred <- predict(modTree,newdata=test)
table(pred,test$classe)

testTree<-predict(modTree,newdata=test)
cmTree<-confusionMatrix(testTree, test$classe)
## accuracy
cmTree$overall[1]

## try a gbm model
modGBM <- train(classe ~ .,method="gbm",data=train, 
      trControl = trainControl(method = "cv", number = 5), verbose=FALSE)
modGBM

testGBM<-predict(modGBM,newdata=test)
cmGBM<-confusionMatrix(testGBM, test$classe)
## Boosting Model Accuracy
cmGBM$overall[1]

## random forest model is most accurate overall on testing set
## use for predicting 20 original test values for project submission
testRF<-predict(modRF, newdata=testingDataClean)
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(testRF)
```

