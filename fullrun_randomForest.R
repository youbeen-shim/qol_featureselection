# Previously on Preprocessing.R ...
# Imported data, labeled and distinguished numeric vs factr
# Normalize - Min-Max Scaling
# Feature Selection - Boruta
# Split - 5-fold 

# saveRDS(folds, "folds.rds")
folds <- readRDS("folds.rds")

### Random Forest ###
# Required packages
library(randomForest)
library(caret)
library(pROC)

# Function to calculate performance metrics
calculate_metrics <- function(actual, predicted, predicted_prob) {
  # Calculate confusion matrix
  cm <- caret::confusionMatrix(predicted, actual, positive = "X1")
  
  # Calculate AUC
  roc_obj <- pROC::roc(actual, predicted_prob[,2])
  auc <- pROC::auc(roc_obj)
  
  # Return metrics
  return(list(
    Accuracy = cm$overall["Accuracy"],
    Sensitivity = cm$byClass["Sensitivity"],
    Specificity = cm$byClass["Specificity"],
    F1 = cm$byClass["F1"],
    AUC = auc
  ))
}

# Simple Random Forest Implementation
simple_rf <- function(train_data, test_data) {
  # Train model
  rf_model <- randomForest(
    eq5d_cat ~ ., 
    data = train_data,
    ntree = 500,
    importance = TRUE
  )
  
  # Make predictions
  pred_class <- predict(rf_model, test_data)
  pred_prob <- predict(rf_model, test_data, type = "prob")
  
  # Calculate metrics
  metrics <- calculate_metrics(test_data$eq5d_cat, pred_class, pred_prob)
  
  return(list(model = rf_model, metrics = metrics))
}

# Random Forest with Hyperparameter Tuning
tuned_rf <- function(train_data, test_data) {
  # Define tuning grid
  tuning_grid <- expand.grid(
    mtry = seq(2, ncol(train_data)-1, by = 2)
  )
  
  # Set up cross-validation
  ctrl <- trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  )
  
  # Train model with tuning
  rf_tuned <- caret::train(
    eq5d_cat ~ .,
    data = train_data,
    method = "rf",
    metric = "ROC",
    tuneGrid = tuning_grid,
    trControl = ctrl,
    importance = TRUE
  )
  
  # Make predictions
  pred_class <- predict(rf_tuned, test_data)
  pred_prob <- predict(rf_tuned, test_data, type = "prob")
  
  # Calculate metrics
  metrics <- calculate_metrics(test_data$eq5d_cat, pred_class, pred_prob)
  
  return(list(model = rf_tuned, metrics = metrics))
}

# Run models for each fold
results_simple <- list()
results_tuned <- list()

for(i in 1:5) {
  # Use current fold as test set
  test_data <- folds[[i]]
  
  # Combine other folds for training
  train_indices <- setdiff(1:5, i)
  train_data <- do.call(rbind, folds[train_indices])
  
  # Run both models
  results_simple[[i]] <- simple_rf(train_data, test_data)
  results_tuned[[i]] <- tuned_rf(train_data, test_data)
}

# Calculate average metrics across folds
get_mean_metrics <- function(results_list) {
  metrics_mat <- do.call(rbind, lapply(results_list, function(x) unlist(x$metrics)))
  colMeans(metrics_mat)
}

# Print results
cat("Simple Random Forest Results:\n")
print(get_mean_metrics(results_simple))
cat("\nTuned Random Forest Results:\n")
print(get_mean_metrics(results_tuned))

# saveRDS(results_simple, "rf_simple.rds")
# saveRDS(results_tuned, "rf_tuned.rds")

### SMOTE ###
library(recipes)
library(themis)
library(tidymodels)

# Function to run RF with SMOTE and evaluate
rf_with_smote <- function(train_data, test_data) {
  # Create and prepare SMOTE recipe
  smote_recipe <- recipe(eq5d_cat ~ ., data = train_data) %>%
    step_dummy(all_nominal_predictors(), -all_outcomes()) %>%
    step_smote(eq5d_cat) %>%
    prep()
  
  # Apply the recipe to get balanced training data
  train_balanced <- bake(smote_recipe, new_data = NULL)
  
  # Apply the recipe to test data (without SMOTE step)
  test_processed <- bake(smote_recipe, new_data = test_data)
  
  # Train RF model on balanced data
  rf_model <- randomForest(
    eq5d_cat ~ ., 
    data = train_balanced,
    ntree = 500,
    importance = TRUE
  )
  
  # Make predictions on test data
  pred_class <- predict(rf_model, test_processed)
  pred_prob <- predict(rf_model, test_processed, type = "prob")
  
  # Calculate metrics
  metrics <- calculate_metrics(test_data$eq5d_cat, pred_class, pred_prob)
  
  return(list(
    model = rf_model, 
    metrics = metrics,
    balanced_data = train_balanced,
    recipe = smote_recipe
  ))
}

# Run cross-validation with SMOTE
results_smote <- list()

for(i in 1:5) {
  # Use current fold as test set
  test_data <- folds[[i]]
  
  # Combine other folds for training
  train_indices <- setdiff(1:5, i)
  train_data <- do.call(rbind, folds[train_indices])
  
  # Run RF with SMOTE
  results_smote[[i]] <- rf_with_smote(train_data, test_data)
}

# Print results
cat("Random Forest with SMOTE Results:\n")
print(get_mean_metrics(results_smote))

# Check class distribution in balanced training data for first fold
print("Class distribution in balanced training data (first fold):")
table(results_smote[[1]]$balanced_data$eq5d_cat)

# Feature importance from the first model
print("Variable importance (from first fold):")
importance(results_smote[[1]]$model)
varImpPlot(results_smote[[1]]$model)

# saveRDS(results_smote, "rf_simple_smote.rds")

### Compare Tuned vs Smote ###
# Function to run RF with SMOTE and evaluate
rf_tuned_with_smote <- function(train_data, test_data) {
  # Create and prepare SMOTE recipe
  smote_recipe <- recipe(eq5d_cat ~ ., data = train_data) %>%
    step_dummy(all_nominal_predictors(), -all_outcomes()) %>%
    step_smote(eq5d_cat) %>%
    prep()
  
  # Apply the recipe to get balanced training data
  train_balanced <- bake(smote_recipe, new_data = NULL)
  
  # Apply the recipe to test data (without SMOTE step)
  test_processed <- bake(smote_recipe, new_data = test_data)
  
  # Train RF model on balanced data
  rf_model <- randomForest(
    eq5d_cat ~ ., 
    data = train_balanced,
    ntree = tuned_params$mtry,
    importance = TRUE
  )
  
  # Make predictions on test data
  pred_class <- predict(rf_model, test_processed)
  pred_prob <- predict(rf_model, test_processed, type = "prob")
  
  # Calculate metrics
  metrics <- calculate_metrics(test_data$eq5d_cat, pred_class, pred_prob)
  
  return(list(
    model = rf_model, 
    metrics = metrics,
    balanced_data = train_balanced,
    recipe = smote_recipe
  ))
}

# Run cross-validation with SMOTE
results_tuned_smote <- list()

for(i in 1:5) {
  # Use current fold as test set
  test_data <- folds[[i]]
  
  # Combine other folds for training
  train_indices <- setdiff(1:5, i)
  train_data <- do.call(rbind, folds[train_indices])
  
  # Run RF with SMOTE
  results_tuned_smote[[i]] <- rf_tuned_with_smote(train_data, test_data)
}

# Print results
cat("Random Forest, tuned, with SMOTE Results:\n")
print(get_mean_metrics(results_tuned_smote))

# Check class distribution in balanced training data for first fold
print("Class distribution in balanced training data (first fold):")
table(results_tuned_smote[[1]]$balanced_data$eq5d_cat)

# Feature importance from the first model
print("Variable importance (from first fold):")
importance(results_tuned_smote[[1]]$model)
varImpPlot(results_tuned_smote[[1]]$model)

### SHAP ###
library(iml)
library(ggplot2)

# Function to perform SHAP analysis on random forest model
rf_shap_analysis <- function(model, data, training_columns) {
  # Ensure data has all necessary columns
  missing_cols <- setdiff(training_columns, colnames(data))
  for(col in missing_cols) {
    data[[col]] <- 0
  }
  
  # Reorder columns to match training data
  data <- data[, training_columns]
  
  # Create predictor object
  predictor <- Predictor$new(
    model = model,
    data = data[, !names(data) %in% "eq5d_cat"],
    y = data$eq5d_cat,
    type = "prob"
  )
  
  # Calculate SHAP values
  shapley <- Shapley$new(
    predictor = predictor,
    x.interest = data[1, !names(data) %in% "eq5d_cat"]  # Reference instance
  )
  
  return(shapley)
}

# Function to generate SHAP summary plot
plot_shap_summary <- function(shapley_values, data) {
  # Extract feature effects
  effects <- shapley_values$results
  
  # Create plot
  ggplot(effects, aes(x = reorder(feature.value, abs(phi)), y = phi)) +
    geom_boxplot() +
    coord_flip() +
    theme_minimal() +
    labs(
      x = "Features",
      y = "SHAP value (impact on model output)",
      title = "Feature Importance based on SHAP values"
    )
}

# Apply SHAP analysis to our best model
best_fold_index <- which.max(sapply(results_smote, function(x) x$metrics$AUC))
best_model <- results_smote[[best_fold_index]]$model
best_data <- folds[[best_fold_index]]

# 
train_dummies <- colnames(results_smote[[best_fold_index]]$balanced_data)

# Run SHAP analysis
shap_values <- rf_shap_analysis(best_model, best_data, train_dummies)

# Generate and print plot
plot_shap_summary(shap_values, best_data)

# Function for comprehensive model interpretation
rf_interpret <- function(model, data, training_columns) {
  # Ensure data has all necessary columns and correct order
  missing_cols <- setdiff(training_columns, colnames(data))
  for(col in missing_cols) {
    data[[col]] <- 0
  }
  data <- data[, training_columns]
  
  # Create predictor object
  predictor <- Predictor$new(
    model = model,
    data = data[, !names(data) %in% "eq5d_cat"],
    y = data$eq5d_cat,
    type = "prob"
  )
  
  # Calculate SHAP values
  shapley <- Shapley$new(
    predictor = predictor,
    x.interest = data[1, !names(data) %in% "eq5d_cat"]
  )
  
  # Feature importance
  importance <- FeatureImp$new(predictor, loss = "ce")
  
  # Feature effects
  effects <- FeatureEffects$new(predictor, 
                                features = names(data)[names(data) != "eq5d_cat"])
  
  # Most important feature analysis
  top_feature <- importance$results$feature[1]
  pdp <- FeatureEffect$new(predictor, feature = top_feature)
  
  # Generate plots
  shap_plot <- plot_shap_summary(shapley, data)
  imp_plot <- plot(importance)
  effects_plot <- plot(effects)
  pdp_plot <- plot(pdp)
  
  return(list(
    shapley = shapley,
    importance = importance,
    effects = effects,
    pdp = pdp,
    plots = list(
      shap = shap_plot,
      importance = imp_plot,
      effects = effects_plot,
      pdp = pdp_plot
    )
  ))
}
# Generate interpretations
interpretations <- rf_interpret(best_model, best_data, train_dummies)

# Print results
print("Feature Importance Results:")
print(interpretations$importance$results)

# Display plots
print(interpretations$plots$shap)
print(interpretations$plots$importance)
print(interpretations$plots$effects)
print(interpretations$plots$pdp)
