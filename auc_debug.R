# 
# Required packages
library(randomForest)
library(caret)
library(pROC)

# Function to plot ROC curves with debugging
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
    roc_obj <- roc(results[[i]]$true_labels, results[[i]]$pred_probs[,2])
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
    lines(roc_obj, col = fold_colors[i])
    
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
    sprintf("Mean ROC (AUC = %.3f ± %.3f)", mean_auc, sd_auc)
  )
  
  legend("bottomright", 
         legend = legend_text,
         col = c(fold_colors, "grey", "red"),
         lty = c(rep(1, 5), 2, 1),
         lwd = c(rep(1, 5), 1, 2))
  
  # Return the calculated values for further analysis if needed
  invisible(list(
    individual_rocs = roc_list,
    mean_coords = data.frame(fpr = fpr_grid, tpr = mean_tpr),
    aucs = aucs,
    mean_auc = mean_auc,
    sd_auc = sd_auc
  ))
}

# Example usage
debug_results <- plot_roc_curves(results_simple)

plot_roc_curves(results_simple)
