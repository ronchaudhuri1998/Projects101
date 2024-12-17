#Assignment BA WITH R Project

# Neural Network
# Load necessary libraries
library(dplyr)
library(neuralnet)
library(caret)
library(nnet)
library(gains)
# Load the dataset
data <- read.csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")


# Selecting relevant columns based on common attrition factors
# For demonstration, we choose a few common factors affecting attrition
data <- data %>% dplyr::select(Attrition, Age, DistanceFromHome, MonthlyIncome, JobSatisfaction, Gender,BusinessTravel, PerformanceRating, RelationshipSatisfaction,
                        EnvironmentSatisfaction, WorkLifeBalance,YearsInCurrentRole, YearsSinceLastPromotion, NumCompaniesWorked)

# Convert categorical columns to factors
data$Attrition <- as.factor(ifelse(data$Attrition == "Yes", 1, 0))
data$Gender <- as.numeric(ifelse(data$Gender == "Male", 1, 0))
data$BusinessTravel <- as.numeric(factor(data$BusinessTravel, levels = c("Non-Travel", "Travel_Rarely", "Travel_Frequently")))

if (!require(corrplot)) install.packages("corrplot")
if (!require(RColorBrewer)) install.packages("RColorBrewer")

# Load the packages
library(corrplot)
library(RColorBrewer)

# Select only numeric columns from the dataset
numeric_data <- data %>% select_if(is.numeric)

# Compute the correlation matrix
correlation_matrix <- cor(numeric_data, use = "complete.obs")

# Create the heatmap using corrplot
corrplot(correlation_matrix, method = "color", 
         col = colorRampPalette(brewer.pal(8, "RdYlBu"))(200), # Color palette
         type = "upper",      # Show only the upper triangle of the matrix
         addCoef.col = "black", # Add correlation values to the plot
         tl.col = "black",    # Color of the variable labels
         tl.srt = 45)         # Rotate the variable labels

# Split data into training and test sets (60-40 split)
set.seed(123)
index <- createDataPartition(data$Attrition, p = 0.6, list = FALSE)
train_data <- data[index, ]
test_data <- data[-index, ]

# Scale numeric features for neural network input
scaled_train <- as.data.frame(scale(train_data %>% select(-Attrition)))
scaled_test <- as.data.frame(scale(test_data %>% select(-Attrition)))
scaled_train$Attrition <- train_data$Attrition
scaled_test$Attrition <- test_data$Attrition

scaled_train$YearsInCurrentRole_PerformanceRating <- scaled_train$YearsInCurrentRole * scaled_train$PerformanceRating
scaled_test$YearsInCurrentRole_PerformanceRating <- scaled_test$YearsInCurrentRole * scaled_test$PerformanceRating



# Fit the neural network model with the manually created interaction term
set.seed(123)
nn_model <- neuralnet(Attrition ~ Age + Gender + DistanceFromHome + MonthlyIncome + JobSatisfaction + WorkLifeBalance +
                        EnvironmentSatisfaction  + YearsInCurrentRole_PerformanceRating + RelationshipSatisfaction +
                        YearsSinceLastPromotion + NumCompaniesWorked + BusinessTravel,
                      data = scaled_train, hidden = c(13, 10, 5), stepmax = 1e8, learningrate = 0.005, linear.output = FALSE)


# Plot the neural network structure
plot(nn_model)

# Step 1: Make predictions on the test data (class labels)
predictions <- predict(nn_model, scaled_test, type = "class");predictions

result_df <- data.frame(scaled_test, Predicted_Attrition = predicted_class)

# View the combined data frame with features and predictions
head(result_df)


# Step 2: Ensure that predictions are factor type with correct levels (0 and 1)
predicted_class <- ifelse(predictions[,1] > 0.5, 0, 1)


# Step 4: Calculate RMSE (Root Mean Squared Error)
predicted_class_numeric <- as.numeric(as.character(predicted_class))
actual_class_numeric <- as.numeric(as.character(scaled_test$Attrition))

# Step 4: Calculate RMSE (Root Mean Squared Error)
rmse <- sqrt(mean((predicted_class_numeric - actual_class_numeric) ^ 2))

# Step 5: Calculate MAE (Mean Absolute Error)
mae <- mean(abs(predicted_class_numeric - actual_class_numeric))

# Print RMSE and MAE
cat("RMSE:", rmse, "\n")
cat("MAE:", mae, "\n")

# Convert to factor with correct levels
predicted_class <- factor(predicted_class, levels = c(0, 1))

# Step 4: Ensure the actual labels are factors with correct levels
actual_class <- factor(scaled_test$Attrition, levels = c(0, 1))

# Step 5: Generate the confusion matrix
conf_matrix <- confusionMatrix(predicted_class, actual_class)
print(conf_matrix)


actual_class <- as.numeric(test_data$Attrition)

library(pROC)
roc_obj <- roc(actual_class, predictions[,1])
plot(roc_obj, main="ROC Curve")
auc(roc_obj)

actual_numeric <- as.numeric(as.character(scaled_test$Attrition))  # Convert actual class to numeric
predicted_probabilities <- predictions[,1]  # Extract probabilities for class 0

rmse <- sqrt(mean((predicted_class - actual_class) ^ 2))

# Step 5: Calculate MAE (Mean Absolute Error)
mae <- mean(abs(predicted_class - actual_class))

# Print RMSE and MAE
cat("RMSE:", rmse, "\n")
cat("MAE:", mae, "\n")

# Step 7: Gains Chart
gains_chart <- gains(actual_numeric, predicted_probabilities, groups = 10)

# Plot the Gains Chart
plot(c(0, gains_chart$cume.pct.of.total * 100) ~ c(0, gains_chart$cume.obs),
     type = "l", col = "blue", main = "Gains Chart (Attrition = 0)", 
     xlab = "Number of Cases", ylab = "Cumulative % of Non-Attrition Responses")
abline(0, max(gains_chart$cume.pct.of.total * 100) / max(gains_chart$cume.obs), lty = 2)
