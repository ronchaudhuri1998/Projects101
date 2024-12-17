rm(list=ls())
# Load necessary libraries
library(rpart)
library(rpart.plot)  # For visualizing the tree
library(caret)       # For calculating model accuracy
library(ggplot2)
library(reshape2)
library(corrplot)
# Read in the data
Campaigns.df <- read.csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
#Campaigns.df<- as.data.frame(Campaigns.df)
# Prepare data: Convert Attrition to binary and remove unnecessary columns
Campaigns.df$Attrition <- ifelse(Campaigns.df$Attrition == "Yes", 1, 0)
CampaignsCut.df <- Campaigns.df[, c(-9, -22, -27)]  # Drop columns as needed
numeric_vars <- CampaignsCut.df[sapply(CampaignsCut.df, is.numeric)]

# Calculate the correlation matrix
cor_matrix <- cor(numeric_vars)
cor_matrix_melt <- melt(cor_matrix)
ggplot(cor_matrix_melt, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0,
                       limit = c(-1, 1), space = "Lab", name="Correlation") +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, size =0.1, hjust = 1)) +
  coord_fixed() +
  ggtitle("Correlation Heatmap of Numeric Variables")+
geom_text(aes(label = round(value, 2)), color = "black", size = 3)

#write.csv(cor_matrix, "correlation_matrix.csv")
# Assuming your dataset is named df
Campaigns.df <- Campaigns.df[, !(names(df) %in% c("YearsAtCompany", "HourlyRate", "EmployeeNumber", "ID", "JobLevel", "DistanceFromHome"))]

#----------------------------------------------------decision tree-------------
# Split data into training and validation sets
set.seed(123)  # Set seed for reproducibility
campaign.index <- sample(1:nrow(CampaignsCut.df), 0.7 * nrow(CampaignsCut.df))
training.df <- CampaignsCut.df[campaign.index, ]
validation.df <- CampaignsCut.df[-campaign.index, ]
#----------------------complete tree--------------------------
# Build the complete decision tree model
campaign.tree <- rpart(Attrition ~ ., data = training.df, method = "class", 
                       control = rpart.control(minbucket = 5, maxdepth =12, cp = 0.01))
# Plot complete tree
rpart.plot(campaign.tree, type = 2, extra = 104, fallen.leaves = TRUE, main = "Decision Tree for Employee Attrition",cex=0.7,box.palette = "RdYlGn")
# Summary of the complete tree model
summary(campaign.tree)
# Displays the complexity parameter table
printcp(campaign.tree)
# Make predictions on the validation set
campaign.tree.pred <- predict(campaign.tree, validation.df, type = "class")
# Evaluate the model's accuracy
confMat<-confusionMatrix(as.factor(campaign.tree.pred), as.factor(validation.df$Attrition))
print(confMat)
# Optional: Use cross-validation for pruning the tree
#---------------------------------------pruned----------------------------------------
selected_cp <-  round(campaign.tree$cptable[which.min(campaign.tree$cptable[, "xerror"]), "CP"], 6)
pruned.tree <- prune(campaign.tree, cp = selected_cp)
# Plot the pruned tree
rpart.plot(pruned.tree,type = 2,extra = 104,fallen.leaves = TRUE,main = "Pruned Decision Tree for Employee Attrition",cex = 1.2,tweak = 0.6,box.palette = "RdYlGn")
# Make predictions with the pruned tree and check accuracy
campaign.pruned.pred <- predict(pruned.tree, validation.df, type = "class")
conf.matrix<-confusionMatrix(as.factor(campaign.pruned.pred), as.factor(validation.df$Attrition))
print(conf.matrix)
##----------------------------------------------------------------------------------------------------------
  ##confusion matrix plots

complete_conf_mat <- as.table(confMat$table)  # Complete tree confusion matrix
# Melt the confusion matrix into long format for ggplot
complete_conf_melt <- as.data.frame(complete_conf_mat)

colnames(complete_conf_melt) <- c("Predicted", "Actual", "Count")
complete_conf_melt$Prediction <- factor(complete_conf_melt$Prediction, levels = c(0, 1))
complete_conf_melt$Reference <- factor(complete_conf_melt$Reference, levels = c(0, 1))
ggplot(data = complete_conf_melt, aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "lightblue", high = "blue", name = "Count") +
  geom_text(aes(label = Count), color = "Red", size = 5) +
  theme_minimal() +
  labs(title = "Confusion Matrix: Complete Tree", 
       x = "Predicted", 
       y = "Actual") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 12),
        axis.text.y = element_text(angle = 45, vjust = 1, size = 12))


pruned_conf_mat <- as.table(conf.matrix$table)  # Complete tree confusion matrix
# Melt the confusion matrix into long format for ggplot
pruned_conf_melt <- as.data.frame(pruned_conf_mat)
colnames(pruned_conf_melt) <- c("Predicted", "Actual", "Count")
pruned_conf_melt$Prediction <- factor(pruned_conf_melt$Prediction, levels = c(0, 1))
pruned_conf_melt$Reference <- factor(pruned_conf_melt$Reference, levels = c(0, 1))
ggplot(data = pruned_conf_melt, aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "lightblue", high = "blue", name = "Count") +
  geom_text(aes(label = Count), color = "Red", size = 5) +
  theme_minimal() +
  labs(title = "Confusion Matrix: pruned Tree", 
       x = "Predicted", 
       y = "Actual") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 12),
        axis.text.y = element_text(angle = 45, vjust = 1, size = 12))


