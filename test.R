library(iml)
library(ggplot2)
library(recipes)
library(themis)
library(tidymodels)
library(randomForest)
library(caret)
library(pROC)
options(future.globals.maxSize = 1100 * 1024^2)

results_smote <- readRDS("rf_simple_smote.rds")
folds <- readRDS("folds.rds")

best_fold_index <- which.max(sapply(results_smote, function(x) x$metrics$AUC))
best_model <- results_smote[[best_fold_index]]$model
best_data <- folds[[best_fold_index]]

train_dummies <- colnames(results_smote[[best_fold_index]]$balanced_data)

missing_cols <- setdiff(train_dummies, colnames(best_data))
for(col in missing_cols) {
  best_data[[col]] <- 0
}
best_data <- best_data[, train_dummies]

predictor <- Predictor$new(
  model = best_model,
  data = best_data[, !names(best_data) %in% "eq5d_cat"],
  y = best_data$eq5d_cat,
  type = "prob"
)

# Calculate SHAP values
shapley <- Shapley$new(
  predictor = predictor,
  x.interest = best_data[1, !names(best_data) %in% "eq5d_cat"]
)

# Feature importance
importance <- FeatureImp$new(predictor, loss = "ce")

# Feature effects
valid_features <- names(which(sapply(best_data[!names(best_data) %in% "eq5d_cat"], 
                                     function(x) length(unique(x))) > 1))
effects <- FeatureEffects$new(predictor, features = valid_features)
# effects <- FeatureEffects$new(predictor, 
#                               features = names(best_data)[names(best_data) != "eq5d_cat"])

# Most important feature analysis
top_feature <- importance$results$feature[1]
pdp <- FeatureEffect$new(predictor, feature = top_feature)

# Generate plots
plot_shap_summary(shapley, best_data)
plot(importance)
plot(effects)
plot(pdp)
