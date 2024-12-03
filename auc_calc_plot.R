folds <- readRDS("folds.rds")

# Required packages
library(randomForest)
library(caret)
library(pROC)
library(ggplot2)

# Modified simple_rf function to store probabilities and true labels
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
  
  # Return everything needed for ROC curves
  return(list(
    model = rf_model, 
    metrics = metrics,
    true_labels = test_data$eq5d_cat,
    pred_probs = pred_prob
  ))
}

# Function to plot ROC curves
plot_roc_curves <- function(results) {
  # Create empty plot
  plot(0:1, 0:1, type = "n", 
       xlab = "False Positive Rate", 
       ylab = "True Positive Rate",
       main = "ROC curve (Random Forest)")
  
  # Add diagonal reference line
  abline(0, 1, lty = 2, col = "grey")
  
  # Colors for individual folds
  fold_colors <- c("#CCCCCC", "#BBBBBB", "#AAAAAA", "#999999", "#888888")
  
  # Store AUCs and ROC objects
  aucs <- numeric(length(results))
  roc_list <- list()
  
  # Plot individual fold ROC curves and store coordinates
  all_coords <- list()
  for(i in 1:length(results)) {
    # Create ROC object
    roc_obj <- roc(results[[i]]$true_labels, results[[i]]$pred_probs[,2], direction = "<")
    roc_list[[i]] <- roc_obj
    aucs[i] <- pROC::auc(roc_obj)
    
    # Store coordinates
    coords <- data.frame(
      specificity = roc_obj$specificities,
      sensitivity = roc_obj$sensitivities
    )
    all_coords[[i]] <- data.frame(
      fpr = 1 - coords$specificity,
      tpr = coords$sensitivity
    )
    
    # Plot ROC curve
    # lines(roc_obj, col = fold_colors[i])
    # Plot ROC curve
    lines(1 - coords$specificity, coords$sensitivity, 
         type = "l", col = fold_colors[i], add = TRUE)
    
    # Print debug info
    cat(sprintf("Fold %d - AUC: %.3f, Points: %d\n", 
                i, aucs[i], nrow(all_coords[[i]])))
  }
  
  # Calculate mean ROC curve
  # First, create a common FPR grid
  fpr_grid <- seq(0, 1, length.out = 100)
  
  # For each FPR point, interpolate TPR values from each fold
  mean_tpr <- sapply(fpr_grid, function(fpr) {
    fold_tprs <- sapply(all_coords, function(coords) {
      # Find closest FPR points and interpolate
      idx <- which.min(abs(coords$fpr - fpr))
      if (length(idx) > 0) {
        return(coords$tpr[idx])
      } else {
        return(NA)
      }
    })
    mean(fold_tprs, na.rm = TRUE)
  })
  
  # Plot mean ROC curve
  lines(fpr_grid, mean_tpr, col = "red", lwd = 2)
  
  # Calculate mean AUC and its standard deviation
  mean_auc <- mean(aucs)
  sd_auc <- sd(aucs)
  
  # Print debug info for mean curve
  cat(sprintf("\nMean AUC: %.3f (± %.3f)\n", mean_auc, sd_auc))
  
  # Add legend
  legend_text <- c(
    sprintf("ROC fold %d (AUC = %.3f)", 0:4, aucs),
    "Chance",
    sprintf("Mean ROC (AUC = %.3f ± %.3f)", mean(aucs), sd(aucs))
  )
  
  legend("bottomright", 
         legend = legend_text,
         col = c(fold_colors, "grey", "red"),
         lty = c(rep(1, 5), 2, 1),
         lwd = c(rep(1, 5), 1, 2),
         cex = 0.8,  # Adjust text size
         inset = c(0.03, 0.03),
         y.intersp = 0.3)
}

# Run the analysis and create plot
results_simple <- list()

for(i in 1:5) {
  # Use current fold as test set
  test_data <- folds[[i]]
  
  # Combine other folds for training
  train_indices <- setdiff(1:5, i)
  train_data <- do.call(rbind, folds[train_indices])
  
  # Run model
  results_simple[[i]] <- simple_rf(train_data, test_data)
}

# Create the plot
plot_roc_curves(results_simple)

