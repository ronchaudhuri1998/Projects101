# Load necessary libraries
library(tidyverse)
library(dplyr)
library(caret)
library(MASS)
library(ggplot2)
library(corrplot)
library(pROC)
library(stringr)
library(GGally)
library(patchwork)
library(Metrics)

# Read the CSV file
data <- read.csv("HR-Employee-Attrition.csv", stringsAsFactors = TRUE)

#DATA EXPLORATION & VISUALIZATION

#Descriptive stats
str(data)
summary(data)


#Histograms

# Create individual plots
plot1 <- ggplot(data, aes(x = Age, fill = Attrition)) + 
  geom_histogram(binwidth = 5, position = "dodge") +
  theme_minimal() +
  labs(
    title = "Attrition across Ages",
    x = "Age",
    y = "Count"
  )

plot2 <- ggplot(data, aes(x = DistanceFromHome, fill = Attrition)) + 
  geom_histogram(binwidth = 5, position = "dodge") +
  theme_minimal() +
  labs(
    title = "Attrition across Distance from Home",
    x = "Distance from home",
    y = "Count"
  )

plot3 <- ggplot(data, aes(x = YearsAtCompany, fill = Attrition)) + 
  geom_histogram(binwidth = 2, position = "dodge") +
  theme_minimal() +
  labs(
    title = "Attrition across Tenures",
    x = "Years at Company",
    y = "Count"
  )

plot4 <- ggplot(data, aes(x = WorkLifeBalance, fill = Attrition)) + 
  geom_histogram(binwidth = 2, position = "dodge") +
  theme_minimal() +
  labs(
    title = "Attrition vs. Work Life Balance",
    x = "Work Life Balance",
    y = "Count"
  )

# Combine the plots using patchwork
combined_plot <- (plot1 | plot2) / (plot3 | plot4)

# Display the combined plot
combined_plot


#Scatter plots
ggplot(data, aes(x = Age, y = MonthlyIncome)) +
  geom_point(alpha = 0.6, color = "blue") +
  geom_smooth(method = "lm", se = FALSE, color = "red", linetype = "dashed") +
  labs(
    title = "Age vs Monthly Income",
    x = "Age",
    y = "Monthly Income"
  ) +
  theme_minimal()

ggplot(data, aes(x = YearsAtCompany, y = TotalWorkingYears)) +
  geom_point(alpha = 0.6, color = "black") +
  geom_smooth(method = "lm", se = FALSE, color = "red", linetype = "dashed") +
  labs(
    title = "Years at Company vs Total Working Years",
    x = "Years At Company",
    y = "Total Working Years"
  ) +
  theme_minimal()

# Correlation heatmap
numeric_vars <- select_if(data, is.numeric)
cor_matrix <- cor(numeric_vars, use = "complete.obs")
corrplot(cor_matrix, method = "color", type = "upper", tl.col = "black")

#DATA CLEANING

#Remove columns with single unique value
colnames(data)
data <- data[ ,c(-22,-9,-10,-26,-27)]
str(data)

# Check for missing values in the dataset
sum(is.na(data))  # Total count of NA values
colSums(is.na(data))  # Check missing values by column


# FEATURE ENGINEERING

#Scaling numerical features
data$NormalizedIncome <- scale(data$MonthlyIncome)

# Binning Age into categories
data$Age_Group <- cut(data$Age, breaks = c(18, 30, 40, 50, 60, 100), 
                      labels = c("18-30", "31-40", "41-50", "51-60", "60+"))


# Feature based on Job satisfaction and Work life balance
data$WorkSatisfaction <- data$JobSatisfaction * data$WorkLifeBalance


# Binary encoding
data$Overtime_Encoded <- ifelse(data$OverTime=="Yes",1,0)
data$Gender_Encoded <- ifelse(data$Gender=="Male",1,0)


#Remove redundant columns
data <- data[-c(1,15,17,20,26)]
str(data)
colnames(data)
ncol(data)

# MODEL BUILDING

set.seed(123)
trainIndex <- createDataPartition(data$Attrition, p = 0.6, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Scaling numeric variables
preProcess_scaler <- preProcess(trainData, method = c("center", "scale"))
trainData <- predict(preProcess_scaler, trainData)
testData <- predict(preProcess_scaler, testData)

# Build the logistic regression model
logit_model <- glm(Attrition ~ ., data = trainData, family = "binomial")
summary(logit_model)

# Use stepwise regression for feature selection
step_model <- stepAIC(logit_model, direction = "both")
summary(step_model)

# Make predictions
predictions <- predict(step_model, testData, type = "response")

# Convert probabilities to binary outcomes
threshold <- 0.5
predicted_classes <- ifelse(predictions > threshold, "Yes", "No")

# Confusion Matrix
confusionMatrix(as.factor(predicted_classes), testData$Attrition)

#ROC curve
roc_curve <- roc(testData$Attrition, predictions)
plot(roc_curve)
auc(roc_curve)

#Lift Chart

# Prepare data for lift chart
lift_data <- data.frame(
  Actual = ifelse(testData$Attrition == "Yes", 1, 0),  # Convert Attrition to binary (1 for "Yes", 0 for "No")
  Predicted = predictions                              # Predicted probabilities from the model
)

# Sort data by predicted probabilities in descending order
lift_data <- lift_data[order(-lift_data$Predicted), ]

# Calculate cumulative actual positives
lift_data$cumulative_positives <- cumsum(lift_data$Actual)

# Add row numbers for cumulative population
lift_data$cumulative_population <- 1:nrow(lift_data)

# Calculate lift and random lift
lift_data$lift <- lift_data$cumulative_positives / sum(lift_data$Actual)
lift_data$random_lift <- lift_data$cumulative_population / nrow(lift_data)

# Plot lift chart
ggplot(lift_data, aes(x = cumulative_population / nrow(lift_data))) +
  geom_line(aes(y = lift, color = "Model Lift"), size = 1) +
  geom_line(aes(y = random_lift, color = "Random Lift"), linetype = "dashed", size = 1) +
  scale_color_manual(values = c("Model Lift" = "blue", "Random Lift" = "red")) +
  labs(
    title = "Lift Chart",
    x = "Cumulative Percentage of Population",
    y = "Cumulative Lift",
    color = "Legend"
  ) +
  theme_minimal()


# Remove rows with NA predictions
valid_rows <- !is.na(predictions)
actual <- testData$Attrition_binary[valid_rows]
predicted <- predictions[valid_rows]

# Recalculate RMSE and MAE
rmse <- sqrt(mean((actual - predicted)^2))
cat("RMSE:", rmse, "\n")

mae <- mean(abs(actual - predicted))
cat("MAE:", mae, "\n")


summary(predictions)
