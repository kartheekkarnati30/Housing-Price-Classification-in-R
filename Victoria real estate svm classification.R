library(ggplot2)
library(dplyr)
library(tidyr)
library(stringr)
library(data.table)
library(rio)
library(modelr)
library(purrr)

getwd()
#Victoria real estate data set:
#Visualizaition:
vic <- fread('C:/Users/karth/Documents/Victoria real estate.csv')
head(vic)
class(vic)
vic <- as_tibble(vic)
class(vic)

vic
vic$postcode <- as.factor(vic$postcode)
vic$parkingSpaces <- as.factor(vic$parkingSpaces)
vic$bathrooms <- as.factor(vic$bathrooms)
vic$bedrooms <- as.factor(vic$bedrooms)
vic$suburb <- as.factor(vic$suburb)

dim(vic)
install.packages("ggpubr")
library(ggpubr)

#Function for cross validation and test set accuracy:
cvlmf <- function(fitt, data, k){
  set.seed(1)
  data <- crossv_kfold(data, k)
  data <- data %>%
    mutate(fit = map(train, ~ fitt, data = .))
  data <- data %>% 
    mutate(rmse_train = map2_dbl(fit, train, ~rmse(.x,.y)),
           rmse_test = map2_dbl(fit, train, ~rmse(.x,.y)) )
  #returning the crossvalidated error on the test sets:
  return(mean(data$rmse_test))
}

dim(na.omit(vic))

#Removing dollar sign from column in price:
vic
vic$price
vic <- vic %>% filter(price != "Contact agent")
vic <- vic %>%select(postcode, region, bedrooms, bathrooms, parkingSpaces, price)
vic_no_miss_region <- vic %>% filter(region!="")
vic_miss_region <- vic %>% filter(region=="")
vic %>% filter(postcode=="")
vic %>% filter(bedrooms=="")
vic %>% filter(bathrooms=="")
vic %>% filter(parkingSpaces=="")
vic %>% filter(price=="")

#using existing values to fill missing spaces in region column:
vic_no_miss_region_np <- vic_no_miss_region %>% select(-price)
vic_no_miss_region_np
vic_miss_region_np <- vic_miss_region %>% select(-price)
vic_miss_region_np

#using nnets to predict regions:
library(nnet)
modelnnet <- nnet::multinom(region ~., data = vic_no_miss_region_np, MaxNWts=4352)
summary(modelnnet)

#predicted regions:
predicted.regions <- modelnnet %>% predict(vic_miss_region_np)

vic_miss_region_np <- cbind(vic_miss_region_np, predicted.regions)
vic_miss_region_np <- vic_miss_region_np %>% select(-region)
names(vic_miss_region_np)[names(vic_miss_region_np) == 'predicted.regions'] <- 'region'
vic_miss_region_np <- as_tibble(vic_miss_region_np)

#regions:
region <- vic_miss_region_np %>% select(region)
#converting region from factor to character:
region[] <- lapply(region, as.character)

vic_miss_region_np <- vic_miss_region_np%>% select(-region)
vic_miss_region_np
vic_miss_region_np <- cbind(vic_miss_region_np, region)
vic_miss_region_np <- as_tibble(vic_miss_region_np)

#combining training and test sets:
data_set <- rbind(vic_no_miss_region_np, vic_miss_region_np)
data_set <- as_tibble(data_set)

#getting price of training and test sets:
price_a <- vic %>% filter(region!="")%>% select(price)
price_b <- vic %>% filter(region=="")%>% select(price)
price <- rbind(price_a, price_b)

#combining price and data_set:
data_set <- cbind(data_set, price)
data_set <- as_tibble(data_set)

#removing dollar sign from price:
strip_dollars = function(x) { as.numeric(gsub("[\\$, ]", "", x)) }
price_sans_dollar <- lapply(data_set[ , c("price")] , strip_dollars) 
data_set <- data_set %>% select(-price)

#combining data_set and price_sans_dollar:
data_set <- cbind(data_set, price_sans_dollar)
data_set <- as_tibble(data_set)
View(data_set)

#converting price into a categorical variable:
price_ranges <- cut(
  data_set$price,
  breaks = c(172500, 376000, 460000, 6100000),
  labels = c("low_end", "mid_range", "high_end"),
  right  = FALSE
)


data_set <- cbind(data_set, price_ranges)
data_set <- as_tibble(data_set)
data_set2 <- data_set %>% select(-price)

#Classification models on data_set2:\
#SVM
data_set3 <- data_set2 %>%filter(!is.na(price_ranges))
library(e1071)

ind <- sample(2,nrow(data_set3),replace=TRUE,prob=c(0.7,0.3))
table(ind)

trainDatasvm <- data_set3[ind==1,]
testDatasvm <- data_set3[ind==2,]
dim(trainDatasvm)

svmtrainx <- trainDatasvm[,-6]
svmtrainy <- trainDatasvm[6]
svmtestx <- testDatasvm[,-6]
svmtesty <- testDatasvm[6]
model_svm_vic <- svm(price_ranges~region+bedrooms+bathrooms+parkingSpaces, data=trainDatasvm)
summary(model_svm_vic)

pred_svm <- predict(model_svm_vic, svmtestx)
library(caret)
confusionMatrix(pred_svm, svmtesty$price_ranges)
#Accuracy of 84.76% on the test set.


#Using kernel SVM model: vanilladot
library(kernlab)
model_vanilladot_svm_vic <- ksvm(price_ranges~region+bedrooms+bathrooms+parkingSpaces, 
                                 data=trainDatasvm, kernel= "vanilladot")
summary(model_vanilladot_svm_vic)
pred_svm_vanilladot <- predict(model_vanilladot_svm_vic, svmtestx)
confusionMatrix(pred_svm_vanilladot, svmtesty$price_ranges)

#Using kernel SVM model: polynomial
model_poly_svm_vic <- ksvm(price_ranges~region+bedrooms+bathrooms+parkingSpaces, 
                                 data=trainDatasvm, kernel= "polydot")
summary(model_poly_svm_vic)
pred_svm_poly <- predict(model_poly_svm_vic, svmtestx)
confusionMatrix(pred_svm_poly, svmtesty$price_ranges)



