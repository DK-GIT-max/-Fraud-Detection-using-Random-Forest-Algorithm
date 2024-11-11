library(dplyr)        # Data manipulation
library(caTools)      # Data splitting
library(caret)        # Model evaluation
library(randomForest) # Random forest model
library(ROSE)         # For random oversampling
library(ggplot2)      # Visualization

# Load the data
data <- read.csv("E:/Desktop/MBA/2nd Year/3rd Semester/AFA/Latest/creditcard.csv")
head(data)

# Display the structure of the dataset
str(data)
summary(data)

# Check for missing values
sum(is.na(data))  # Should return 0 if there are no missing values

# Check the class distribution
ggplot(data, aes(x = Class)) + geom_bar()

# Split the data into training and testing sets (70% training, 30% testing)
set.seed(123)  # For reproducibility
split <- sample.split(data$Class, SplitRatio = 0.7)

train_data <- subset(data, split == TRUE)
test_data <- subset(data, split == FALSE)
sum(train_data$Class)  # Check distribution in training data

# Apply ROSE to balance the training data (deals with class imbalance)
train_data_rose <- ROSE(Class ~ ., data = train_data, seed = 123)$data

# Check the class distribution after applying ROSE
table(train_data_rose$Class)

# Convert the 'Class' column to a factor for classification
train_data_rose$Class <- as.factor(train_data_rose$Class)
test_data$Class <- as.factor(test_data$Class)

# Train the Random Forest model on the ROSE balanced data
set.seed(123)
rf_model <- randomForest(Class ~ ., data = train_data_rose, ntree = 50, importance = TRUE)

# Make predictions on the test data
rf_predictions <- predict(rf_model, newdata = test_data)

# Create a confusion matrix to evaluate model performance
cm <- confusionMatrix(rf_predictions, test_data$Class)
print(cm)

# Feature importance scores from Random Forest
importance(rf_model)

# Implementing k-fold cross-validation to get a more robust estimate of model performance
train_control <- trainControl(method = "cv", number = 10)  # 10-fold CV
model <- train(Class ~ ., data = train_data_rose, method = "rf", trControl = train_control)
print(model)

# To extract accuracy or other performance metrics from the cross-validation
model$results
