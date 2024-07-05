library(dendextend)
library(dbscan)
library(ggbiplot)
library(ggm)
library(limma)
library(tidyverse)
library(tibble)
library(dplyr)

################################ Clustering function ###################################################################
ClusterOptim <- function(file_name, classes, plotAll = FALSE, outDir = "Clusters") {
  # Create table for summary data
  acc_all_optimal_PCs <- tibble(Sample_removed = "-", PCs = "NA", Accuracy = 0, Sensitivity = 0, Specificity = 0, MRD_best_cut = 0, P_value = 0)

  # Label with file details
  label <- strsplit2(file_name, "/")
  label <- label[length(label)]
  label <- strsplit2(label, ".", fixed = T)[1]

  # Read file
  orig.df <- read.csv(file_name, row.names = 1, header = T, stringsAsFactors = F, check.names = F)


  # Remove any samples with no reads and rows which sum to 0
  orig.df <- orig.df[, apply(orig.df, MAR = 2, function(x) sum(x) > 0)]
  orig.df <- orig.df[apply(orig.df, 1, function(x) sum(x) != 0), ]

  # Normalise kmer counts by total number of kmers
  orig.df <- apply(orig.df, MARGIN = 2, function(x) x / sum(x, na.rm = TRUE))

  # transpose data for PCA
  t.df <- t(orig.df)

  # Define coeliac and normal samples
  tags <- row.names(t.df)
  colours <- c("blue", "red")
  # classes<-integer(length(tags))#now we find the classes
  ones <- which(classes == 1) # Coeliacs are 1s
  twos <- which(classes == 2) # Normals are 2s
  # classes[ones]<-1
  # classes[twos]<-2

  if (length(classes) != ncol(orig.df)) {
    print("Error: different number of samples and true classes")
    return(0)
  }

  # Perform PCA
  PCA <- prcomp(t.df, center = T, scale = F)
  pca_data <- PCA$x
  # convert to tibble
  pct <- as_tibble(pca_data)

  # PCA combinations to be analysed
  ranges <- powerset(1:10, nonempty = T)

  # Generate tibble to store accuracy data
  PC_acc <- tibble(PCs = "NA", Accuracy = 0, Sensitivity = 0, Specificity = 0, MRD_best_cut = 0, P_value = 0, adj_P_value_BH = 0, adj_P_value_bonferroni = 0)

  # Cycle through each PC combination and assess clustering efficiency
  # output_file
  if (plotAll) {
    pdf(file = paste0(outDir, "/", label, "_ALL_PC_combinations.pdf"))
  }

  for (i in 1:length(ranges)) { # 1:length(ranges)从1开始循环，直到达到ranges的长度
    PCs <- unlist(ranges[i])

    # Cluster samples and create dendrogram obecjt
    clusterer <- hdbscan(pct[, PCs], minPts = 5, gen_hdbscan_tree = T)
    dendro <- clusterer$hdbscan

    # reorder dendrogram
    dendro <- reorder(dendro, length(tags):1) # length(tags):1创建一个从1开始，长度与tags相同的向量

    # set sample attributes and colour accordingly
    classtags <- classes[labels(dendro)]
    nametags <- tags[labels(dendro)]
    coltags <- colours[classtags]
    labels_colors(dendro) <- coltags
    labels(dendro) <- nametags

    ####### SCORE THE DENDROGRAMS
    nodes <- get_nodes_attr(dendro, "label")
    leaves <- which(get_nodes_attr(dendro, "leaf")) # Find which nodes are leaves
    parents <- is.na(get_nodes_attr(dendro, "label")) # Nodes that are nt leaves
    children <- partition_leaves(dendro)
    ## Now we have all the partitioning
    ## We want to find the best line of separation.
    ## There are N-2 possible partitions in our set of N samples
    partitions <- children[parents][-1]
    quality <- matrix(0L, nrow = 2, ncol = length(partitions))
    ## quality measures the quality of the left and right sides of each cut
    left <- right <- c()

    ## Look at each possible cut
    for (j in 1:length(partitions)) {
      left <- setdiff(tags, partitions[[j]])
      left_no_of_CeD <- length(grep("C", left))
      left_no_of_norm <- length(grep("N", left))

      right <- partitions[[j]] # points in the cut
      right_no_of_CeD <- length(grep("C", right))
      right_no_of_norm <- length(grep("N", right))

      if (right_no_of_CeD >= left_no_of_CeD) {
        quality[2, j] <- right_no_of_CeD
        quality[1, j] <- left_no_of_norm
      } else {
        quality[2, j] <- left_no_of_CeD
        quality[1, j] <- right_no_of_norm
      }
    }

    # Identify position of best cut
    q <- apply(X = quality, MARGIN = 2, FUN = sum) # q is the total quality of each cut
    score <- max(q) # score is the best score
    pos <- which(q == score)

    # Get sensitivity, specificity and MRD between at best cut
    sens <- quality[2, pos]
    Sensitivity <- sens
    max.sens.pos <- which(sens == max(sens))
    spec <- quality[1, pos][max.sens.pos]
    MRD_best_cut <- get_branches_heights(dendro, sort = F)[pos] - get_branches_heights(dendro, sort = F)[pos + 1]

    if (plotAll) {
      # Create plot
      plot(dendro, ylab = "Mutual Reachability (Distance)", xlab = label, main = paste0("PCs=", paste(PCs, collapse = ", "), ". Accuracy=", score, "/", length(tags)))
    }
    # Write scores to table
    PC_acc[i, 1] <- paste(PCs, collapse = ", ")
    PC_acc[i, 2] <- score
    PC_acc[i, 3] <- sens[max.sens.pos][1] # Need to put ones here in case of equal top values
    PC_acc[i, 4] <- spec[1]
    PC_acc[i, 5] <- max(MRD_best_cut[max.sens.pos])[1]

    # Calculate P values and add to table
    tp <- sens[max.sens.pos][1]
    tn <- spec[1]
    fp <- length(classes[twos]) - tn
    fn <- length(classes[ones]) - tp
    tab <- matrix(data = NA, nrow = 2, ncol = 2)
    tab[1:2, 1] <- c(tp, fn)
    tab[1:2, 2] <- c(fp, tn)
    ft <- fisher.test(tab)
    PC_acc[i, 6] <- ft$p.value
  }
  if (plotAll) {
    dev.off()
  }

  # Adjust P values to correct for multiple testing
  PC_acc[, 7] <- p.adjust(PC_acc$P_value, method = "BH")
  PC_acc[, 8] <- p.adjust(PC_acc$P_value, method = "bonferroni")

  # Save summary table
  PC_acc <- arrange(PC_acc, desc(Accuracy), desc(Sensitivity), desc(MRD_best_cut))
  write.csv(PC_acc, paste0(outDir, "/", label, "_ALL_PCs_Score_Sens_spec_MRD.csv"))

  # Create accuracy summary table
  acc_table <- table(PC_acc[, 2])
  acc_table_df <- tibble(Score = paste0(names(acc_table), "//", length(tags)), Number_of_PCs = acc_table)
  Score <- paste0(names(acc_table), "//", length(tags))
  acc_table_df <- arrange(acc_table_df, desc(Score))
  write.csv(acc_table_df, paste0(outDir, "/", label, "_ALL_PCs_accuracy_summary.csv"))




  ################## Identify optimal PCs and extract optimal dendrogram plot

  optimal_PCs <- PC_acc[1, 1]

  pdf(file = paste0(outDir, "/", label, "_OPTIMAL_PCs_", optimal_PCs, ".pdf"))
  PCs <- as.double(strsplit2(optimal_PCs, ", "))
  clusterer <- hdbscan(pct[, PCs], minPts = 5, gen_hdbscan_tree = T)
  dendro <- clusterer$hdbscan
  ###### reordering dendrogram
  dendro <- reorder(dendro, length(tags):1)

  # set sample attributes and colour accordingly
  classtags <- classes[labels(dendro)]
  nametags <- tags[labels(dendro)]
  coltags <- colours[classtags]
  labels_colors(dendro) <- coltags
  labels(dendro) <- nametags

  ####### SCORE THE DENDROGRAMS
  nodes <- get_nodes_attr(dendro, "label")
  leaves <- which(get_nodes_attr(dendro, "leaf")) # Find which nodes are leaves
  parents <- is.na(get_nodes_attr(dendro, "label")) # Nodes that are nt leaves
  children <- partition_leaves(dendro)
  ## Now we have all the partitioning
  ## We want to find the best line of separation.
  ## There are N-2 possible partitions in our set of N samples
  partitions <- children[parents][-1]
  quality <- matrix(0L, nrow = 2, ncol = length(partitions))
  ## quality measures the quality of the left and right sides of each cut
  left <- right <- c()

  ## Look at each possible cut
  for (j in 1:length(partitions)) {
    left <- setdiff(tags, partitions[[j]])
    left_no_of_CeD <- length(grep("C", left))
    left_no_of_norm <- length(grep("N", left))

    right <- partitions[[j]] # points in the cut
    right_no_of_CeD <- length(grep("C", right))
    right_no_of_norm <- length(grep("N", right))

    if (right_no_of_CeD >= left_no_of_CeD) {
      quality[2, j] <- right_no_of_CeD
      quality[1, j] <- left_no_of_norm
    } else {
      quality[2, j] <- left_no_of_CeD
      quality[1, j] <- right_no_of_norm
    }
  }

  # Identify best cut
  q <- apply(X = quality, MARGIN = 2, FUN = sum) # q is the total quality of each cut
  score <- max(q) # score is the best score

  ## now do the plotting
  plot(dendro, ylab = "Mutual Reachability (Distance)", xlab = label, main = paste0("PCs=", paste(PCs, collapse = ", "), ". Accuracy=", score, "/", length(tags)))
  dev.off()

  # return score summary to bring together results of all input files
  colnames(acc_table_df)[2] <- label
  acc_table_df
}
################################################################################################################################################
