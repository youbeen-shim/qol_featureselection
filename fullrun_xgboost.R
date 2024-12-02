# Required packages
library(xgboost)
library(caret)
library(pROC)
library(recipes)
library(themis)
library(SHAPforxgboost)
library(ggplot2)

# Utility function for metrics (reused from your existing code)
calculate_metrics <- function(actual, predicted, predicted_prob) {
  cm <- confusionMatrix(predicted, actual) #, positive = "X1")
  
  # Calculate AUC
  roc_obj <- roc(actual, predicted_prob[,2])
  auc <- auc(roc_obj)
  
  return(list(
    Accuracy = cm$overall["Accuracy"],
    Sensitivity = cm$byClass["Sensitivity"],
    Specificity = cm$byClass["Specificity"],
    F1 = cm$byClass["F1"],
    AUC = auc
  ))
}

# XGBoost with SMOTE implementation
xgb_with_smote <- function(train_data, test_data, params = NULL) {
  # Default parameters optimized for speed/performance balance
  if (is.null(params)) {
    params <- list(
      objective = "binary:logistic",
      eval_metric = "auc",
      eta = 0.1,
      max_depth = 6,
      min_child_weight = 1,
      subsample = 0.8,
      colsample_bytree = 0.8
    )
  }
  
  # Prepare SMOTE recipe
  smote_recipe <- recipe(eq5d_cat ~ ., data = train_data) %>%
    step_dummy(all_nominal_predictors(), -all_outcomes(), one_hot = TRUE) %>%
    step_smote(eq5d_cat) %>%
    prep()
  
  # Apply recipe
  train_balanced <- bake(smote_recipe, new_data = NULL)
  test_processed <- bake(smote_recipe, new_data = test_data)
  
  # Prepare matrices for XGBoost
  dtrain <- xgb.DMatrix(
    data = as.matrix(train_balanced[, !names(train_balanced) %in% "eq5d_cat"]),
    label = as.numeric(train_balanced$eq5d_cat == "X1")
  )
  
  dtest <- xgb.DMatrix(
    data = as.matrix(test_processed[, !names(test_processed) %in% "eq5d_cat"])
  )
  
  # Train model with early stopping
  xgb_model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = 100,
    early_stopping_rounds = 10,
    watchlist = list(train = dtrain),
    verbose = 0
  )
  
  # Make predictions
  pred_prob <- predict(xgb_model, dtest)
  pred_class <- factor(ifelse(pred_prob > 0.5, "X1", "X0"), levels = c("X0", "X1"))
  
  # Format probability matrix for metric calculation
  prob_matrix <- cbind(1 - pred_prob, pred_prob)
  colnames(prob_matrix) <- c("X0", "X1")
  
  # Calculate metrics
  metrics <- calculate_metrics(test_data$eq5d_cat, pred_class, prob_matrix)
  
  return(list(
    model = xgb_model,
    metrics = metrics,
    recipe = smote_recipe,
    feature_names = colnames(train_balanced[, !names(train_balanced) %in% "eq5d_cat"])
  ))
}

# Run cross-validation
set.seed(42)  # for reproducibility
results <- list()
cv_metrics <- matrix(NA, nrow = 5, ncol = 5)  # Store metrics for each fold
colnames(cv_metrics) <- c("Accuracy", "Sensitivity", "Specificity", "F1", "AUC")

for(i in 1:5) {
  # Prepare train/test split
  test_data <- folds[[i]]
  train_indices <- setdiff(1:5, i)
  train_data <- do.call(rbind, folds[train_indices])
  
  # Run model
  results[[i]] <- xgb_with_smote(train_data, test_data)
  cv_metrics[i,] <- unlist(results[[i]]$metrics)
}

# Print average results
cat("\nCross-validation results:\n")
print(colMeans(cv_metrics))
print(apply(cv_metrics, 2, sd))

# SHAP analysis on best model
best_fold <- which.max(cv_metrics[,"AUC"])
best_model <- results[[best_fold]]$model
best_data <- folds[[best_fold]]

# Calculate SHAP values
shap_values <- shap.values(
  xgb_model = best_model,
  X_train = as.matrix(bake(results[[best_fold]]$recipe, 
                           new_data = best_data)[, !names(best_data) %in% "eq5d_cat"])
)

# Plot SHAP summary
shap_long <- shap.prep(
  shap_contrib = shap_values$shap_score,
  X_train = as.matrix(bake(results[[best_fold]]$recipe, 
                           new_data = best_data)[, !names(best_data) %in% "eq5d_cat"]),
  top_n = 20  # Limit to top 20 features for clarity
)

shap.plot.summary(shap_long)