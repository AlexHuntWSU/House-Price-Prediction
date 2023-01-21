data = read.csv("train.csv")
library("dplyr") 
library(tidyverse)
library("ggplot2")
library("ggcorrplot")
library("caret")
library("data.table")
library(mltools)
library(pls)
library(glmnet)
library(coefplot)
library(Hmisc)
library(rcompanion)

set.seed(1)
#colSums(is.na(data))[colSums(is.na(data)) > 0]

garage_cols <- c('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond')
bsmt_cols <- c('BsmtExposure', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtFinType1')

data$MiscFeature[is.na(data$MiscFeature)] = 'NA'
data$Alley[is.na(data$Alley)] = 'NA'
data$PoolQC[is.na(data$PoolQC)] = 'NA'
data$Fence[is.na(data$Fence)] = 'NA'
data[garage_cols][is.na(data$GarageType),] = 'NA'
data$Fence[is.na(data$Fence)] = 'NA'
data$GarageYrBlt <- ifelse(is.na(data$GarageYrBlt), data$YearBuilt, data$GarageYrBlt) #Set to YearBuilt
data$FireplaceQu[is.na(data$FireplaceQu)] = 'NA'
data$LotFrontage[is.na(data$LotFrontage)] = 0
data[bsmt_cols][is.na(data$BsmtFinType2),] = 'NA'
data[bsmt_cols][is.na(data$BsmtExposure),] = 'No' #Most common value
data$MasVnrType[is.na(data$MasVnrType)] = 'NA'
data$MasVnrArea[is.na(data$MasVnrArea)] = 0
data$Electrical[is.na(data$Electrical)] = 'SBrkr' #House built in 2006, newer electicity type

#colSums(is.na(train_df))[colSums(is.na(train_df)) > 0]

categorical <- dplyr::select(data, where(is.character) | Id) #Get the categorical data + id for merging later
ordinal <- subset(data,select=c('GarageQual','GarageCond','PoolQC','FireplaceQu',
                                'KitchenQual','HeatingQC','BsmtExposure','BsmtCond',
                                'BsmtQual','ExterCond','ExterQual'))

#Assign a number scale to the ordinal variables
y = colnames(ordinal)
data[y][data[y] == 'NA'] = 0
data[y][data[y] == 'Po'] = 1
data[y][data[y] == 'No'] = 1
data[y][data[y] == 'Mn'] = 2
data[y][data[y] == 'Av'] = 3
data[y][data[y] == 'Fa'] = 2
data[y][data[y] == 'TA'] = 3
data[y][data[y] == 'Gd'] = 4
data[y][data[y] == 'Ex'] = 5
data[y] <- sapply(data[y], as.numeric)

numeric <- select_if(data, is.numeric) 

#Get the rest of the categorical data and make them factors
nominal = categorical[,!names(categorical) %in% names(ordinal)]
nominal[sapply(nominal, is.character)] <- lapply(nominal[sapply(nominal, is.character)], 
                                                 as.factor)
#One hot encoding of nominal variables. Note: Dropping one factor won't be required for the regression models
newdata <- one_hot(as.data.table(nominal))
newdata = as.data.frame(newdata)

#Combine the two data tables
All <- merge(newdata,numeric,by="Id")
#Get columns with near zero variance, provides no value
zero = nearZeroVar(All, names = TRUE)
All = All[,!names(All) %in% zero]

par(mfrow=c(1,2))
hist(All$SalePrice)
hist(log(All$SalePrice))

set.seed(1)
Transformations = c("YearBuilt", "GarageYrBlt", "YearRemodAdd", "GrLivArea", "TotalBsmtSF",
                    "X1stFlrSF", "LotArea", "LotFrontage", "BsmtFinSF1", "BsmtUnfSF", "MasVnrArea")
for (x in Transformations){
  All[,x] = transformTukey(All[,x])
}
All$SalePrice = log(All$SalePrice)


#split data into a test and training set
train <- All %>% dplyr::sample_frac(0.70)
test  <- dplyr::anti_join(All, train, by='Id')

test = test[,-1]
train = train[,-1]

correlation = cor(train, train$SalePrice, method = "pearson")
names = rownames(correlation)
abs_cor = abs(correlation) #absolute value of the correlation coefficients
corr_data = data.frame(variable = names, abs_cor = abs_cor,cor = correlation)
corr_data = corr_data[order(corr_data$abs_cor,decreasing = TRUE),]
head(corr_data, 21)
cols = corr_data[c(1:21),]$variable

#Keep only the top 20 correlated values
train_1 = train[,names(train) %in% cols]
test_1 = test[,names(test) %in% cols]

#Removing Collinearity (>= 0.7)

#1. Combine correlated quality variables
train_1$Quality = train_1$OverallQual + train_1$ExterQual + train_1$KitchenQual + train_1$BsmtQual
#2. Combine interior square footage variables
train_1$SqFt = train_1$GrLivArea + train_1$TotalBsmtSF
#3. Combine year
train_1$Year = (train_1$YearBuilt + train_1$GarageYrBlt)/2
#4. Drop columns
high_corr = c("GarageArea", "FireplaceQu", "TotalBsmtSF", "Foundation_PConc", "GarageYrBlt", "TotRmsAbvGrd", "OverallQual", "ExterQual", "BsmtQual", "KitchenQual", "GrLivArea", "X1stFlrSF", "CentralAir_N", "YearBuilt")
train_1 = train_1 %>% dplyr::select(-SalePrice,SalePrice)
train_1 = train_1[,!names(train_1) %in% high_corr]

test_1$Quality = test_1$OverallQual + test_1$ExterQual + test_1$KitchenQual + test_1$BsmtQual
test_1$SqFt = test_1$GrLivArea + test_1$TotalBsmtSF
test_1$Year = (test_1$YearBuilt + test_1$GarageYrBlt)/2
test_1 = test_1 %>% dplyr::select(-SalePrice,SalePrice)
test_1 = test_1[,!names(test_1) %in% high_corr]

corr <- round(cor(train_1), 1)
ggcorrplot(corr, lab = T, type = "full")


lm.fit = lm(SalePrice ~ ., data=train_1)
summary(lm.fit)
plot(lm.fit)

rmse = mean((test_1$SalePrice - predict(lm.fit, test_1[,-11]))^2) %>% sqrt()
rmse

pcr_model <- pcr(SalePrice ~ ., data=train, scale=TRUE, validation="CV")
comps <- RMSEP(pcr_model)$val[1,,]
best <- which.min(comps) - 1
pred <- predict(pcr_model, test[,-111], ncomp=best)
summary(pcr_model)

#validationplot(pcr_model, val.type = "R2")
rmse <- mean((test$SalePrice - pred)^2) %>% sqrt()
rmse

y = data.matrix(train$SalePrice)
x = data.matrix(train[,-111])
x2 = data.matrix(test[,-111])

cv_model <- cv.glmnet(x, y, alpha = 1) #cross validation to find lowest test error
best_lambda <- cv_model$lambda.min
best_model <- glmnet(x, y, alpha = 1, lambda = best_lambda)
best_model
pred<- predict(best_model, s = best_lambda, newx = x2)
rmse <- mean((test$SalePrice - pred)^2) %>% sqrt()
rmse

par(mfrow=c(1,1))
plot(pred,test$SalePrice,
     xlab = "Predicted Values",
     ylab = "Observed Values")
abline(a = 0, b = 1, lwd=2,
       col = "green")