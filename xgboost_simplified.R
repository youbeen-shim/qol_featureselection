# XGBoost with SMOTE implementation
xgb_with_smote <- function(train_data, test_data) {
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
    params = list(
      objective = "binary:logistic",
      eval_metric = "auc",
      eta = 0.1,
      max_depth = 6,
      min_child_weight = 1,
      subsample = 0.8,
      colsample_bytree = 0.8
    ),
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
    recipe = smote_recipe
  ))
}

# Run cross-validation
set.seed(42)
results_xgb <- list()
results_xgb_smote <- list()

for(i in 1:5) {
  # Prepare train/test split
  test_data <- folds[[i]]
  train_indices <- setdiff(1:5, i)
  train_data <- do.call(rbind, folds[train_indices])
  
  # Run XGBoost without SMOTE
  results_xgb[[i]] <- xgb_with_smote(train_data, test_data)
  
  # Run XGBoost with SMOTE
  results_xgb_smote[[i]] <- xgb_with_smote(train_data, test_data)
}

# Print average results
cat("\nXGBoost Results (without SMOTE):\n")
print(colMeans(do.call(rbind, lapply(results_xgb, function(x) unlist(x$metrics)))))

cat("\nXGBoost Results (with SMOTE):\n")
print(colMeans(do.call(rbind, lapply(results_xgb_smote, function(x) unlist(x$metrics)))))