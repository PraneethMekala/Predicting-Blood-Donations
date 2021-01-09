"Part A: Predicting Blood Donation"
install.packages("car")
install.packages("caret")
library(tidyverse)
library(MASS)
library(car)
library(caret)
library(ggcorrplot)
library(corrplot)
library(caTools)
library(readxl)
library(pROC)
setwd('~/Desktop/Data Science bootcamp/Week 6/Project 06')

df_train = read_excel('blood_traindata.xlsx')
str(df_train)
df_test = read_excel('blood_testdata.xlsx')

#Logistic model
?lm()
log01 = lm(Made Donation this month ~. -ID, data = df_train) #change colnames
?colnames
colnames(df_train) <- c("ID", "Mo.last.donation", "No.donations", "Total.donated", "Mo.first.donation", "donate")
colnames(df_test) <- c("ID", "Mo.last.donation", "No.donations", "Total.donated", "Mo.first.donation")

log01 = glm(donate ~. -ID, data=df_train) #ID is not important to determine donated status
summary(log01) # seems to be an issue in Total.donated

which(rowSums(is.na(df_train))==ncol(df_train)) #find the rows with possible NA values. None

corrplot(cor(df_train, use="complete.obs"), method="number",type="lower") #total donation is corr with number of donations

log02 = glm(donate ~. -ID -Total.donated, data=df_train)
summary(log02)

model01 = stepAIC(log02, direction="both")
summary(model01)

#Other ML models
##
## 10-fold Cross-Validation
##
df_train$ID = NULL
df_train$donate = as.factor(df_train$donate)
str(df_train)
?trainControl

#cross validation <-10 fold
control = trainControl(method="cv", number=10)
metric = "Accuracy"

# Linear Discriminant Analysis (LDA)
fit.lda = train(donate~.-Total.donated, data=df_train, method="lda", metric=metric, trControl=control) 

# Classfication and Regression Trees (CART)
fit.cart = train(donate~., data=df_train, method="rpart", metric=metric, trControl=control)

# k-Nearest Neighbors (KNN)
fit.knn = train(donate~., data=df_train, method="knn", metric=metric, trControl=control)

# Bayesian Generalized Linear Model 
fit.logi = train(donate~., data=df_train, method="bayesglm", metric=metric, trControl=control)

# Support Vector Machines (SVM) --> a long long time
fit.svm = train(donate~., data=df_train, method="svmRadial", metric=metric, trControl=control)

# Random Forest
fit.rf = train(donate~., data=df_train, method="rf", metric=metric, trControl=control)

# Gradient Boosting Machines/XGBoost
fit.xgb = train(donate~., data=df_train, method="xgbLinear", metric=metric, trControl=control)

# Select Best Model
# summarize accuracy of models
results = resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, logi=fit.logi, svm=fit.svm, rf=fit.rf, xgb=fit.xgb)) 
summary(results)

# Summarize the Best Model
print(fit.cart)
attributes(fit.cart)

#make a prediction
predict1 = predict(fit.cart, df_test)
df_test$predicted_donation = predict1

write_csv(df_test, "P6A predictions.csv") #exporting predictions on test to csv file

#Find value of AUC and optimal threshold
pred = predict(fit.xgb, type = "prob", df_train)
pred.1 = as.numeric(pred[,2])
xgb.roc = roc(response = df_train$donate, predictor = pred.1)
plot(xgb.roc, legacy.axes = TRUE, print.auc.y = 1.0, print.auc = TRUE)
coords(xgb.roc, "best", "threshold", transpose = TRUE)
"AUC is 0.97 with an optimal threshold of 0.3769"
