library(limma)
library(tidyverse)
library(data.table)
library(dplyr)


################# FUNCTION TO SPLIT CDR3 INTO NON-POSITIONAL KMERS AND GET COUNTS ################# #################
split_cdr3_into_non_positional_kmers <- function(r, k, sample_names) {
  # Vector of CDR3 counts in each sample
  cdr3_count <- as.integer(unlist(r[-1]))

  # CDR3 sequence to split into kmers
  cdr3 <- r[1]

  # No of kmers for given CDR3
  no_of_kmers <- nchar(cdr3) + 1 - k

  # List of kmers
  kmer_list <- substr(rep(cdr3, no_of_kmers), 1:no_of_kmers, k:nchar(cdr3))

  # Matrix of kmers and counts
  kmer.mtx <- matrix(data = cdr3_count, nrow = no_of_kmers, ncol = length(cdr3_count), byrow = T)
  rownames(kmer.mtx) <- kmer_list

  # Convert to data.table and set column names
  kmer.mtx <- as.data.table(kmer.mtx, keep.rownames = T)
  colnames(kmer.mtx) <- c("kmer", sample_names)

  kmer.mtx
}
################# ################# ################# ################# ################# #################



################# FUNCTION TO SPLIT CDR3 INTO POSITIONAL KMERS AND GET COUNTS ################# #################
split_cdr3_into_positional_kmers <- function(r, k, sample_names, pos = c("start", "middle", "end")) {
  # Vector of CDR3 counts in each sample
  cdr3_count <- as.integer(unlist(r[-1])) # as.double(unlist(r)) #cdr3_count <- as.double(unlist(kmer.list[1]))

  # CDR3 sequence to split into kmers
  cdr3 <- r[1]

  # No of non-positional kmers for given CDR3
  no_of_kmers <- nchar(cdr3) + 1 - k

  # Position kmers into occurring at start, middle of end of CDR3 and count
  l_cut <- nchar(cdr3) / 3
  u_cut <- (2 * nchar(cdr3)) / 3
  kmer_list <- substr(rep(cdr3, no_of_kmers), 1:no_of_kmers, k:nchar(cdr3))

  left <- pmin(k, pmax(l_cut - 1:no_of_kmers + 1, 0))
  middle <- pmin(k, pmax(u_cut - 1:no_of_kmers + 1, 0)) - pmin(k, pmax(l_cut - 1:no_of_kmers + 1, 0))
  right <- k - pmin(k, pmax(u_cut - 1:no_of_kmers + 1, 0))

  prop_left <- left / (left + middle + right)
  prop_middle <- middle / (left + middle + right)
  prop_right <- right / (left + middle + right)

  prop_left <- round(prop_left)
  prop_middle <- round(prop_middle)
  prop_right <- round(prop_right)

  props <- c(prop_left, prop_middle, prop_right) # is arranged by pos, then kmer

  counts <- lapply(as.list(props), function(x) {
    out <- x * cdr3_count
    out
  })

  # Matrix of kmers and counts
  kmer.mtx <- matrix(data = unlist(counts), nrow = 3 * length(kmer_list), ncol = length(cdr3_count), byrow = T)
  rownames(kmer.mtx) <- unlist(lapply(pos, function(p) {
    kp <- paste0(kmer_list, as.list(p))
    kp
  }))

  # Convert to data.table and set column names
  kmer.mtx <- as.data.table(kmer.mtx, keep.rownames = T)
  colnames(kmer.mtx) <- c("kmer", sample_names)

  # Remove rows which sum to 0
  kmer.mtx <- kmer.mtx[rowSums(kmer.mtx[, -1]) > 0, ]

  kmer.mtx
}
############################################################################################################



#### Wrapper functions#####

NonPos_Matrix <- function(InputFile, k, OutputFile) {
  # Input csv should have samples as columns and CDR3 sequences as rows, i.e:
  DT <- fread(InputFile)
  sample.names <- colnames(DT[, -1, ])

  # Remove CDR3 sequences shorter than kmer lenth
  DT <- DT[nchar(kmer) >= k, , ]

  # transpose so can do function on columns
  DT <- transpose(DT)

  # For each CDR3 sequence, identify kmers and associated counts
  NP_DT <- DT[, list(lapply(.SD, FUN = split_cdr3_into_non_positional_kmers, k = k, sample_names = sample.names)), ]

  # Merge, sum, sort and write
  NP_DT <- NP_DT$V1 %>% bind_rows(.)
  NP_DT <- NP_DT[, lapply(.SD, FUN = sum), by = kmer]
  NP_DT <- NP_DT[order(kmer), , ]
  fwrite(NP_DT, OutputFile) # Expects Output file to be csv.gz
  return(NP_DT)
}




Pos_Matrix <- function(InputFile, k, OutputFile) {
  # Input csv should have samples as columns and CDR3 sequences as rows, i.e:
  DT <- fread(InputFile)
  sample.names <- colnames(DT[, -1, ])

  # Remove CDR3 sequences shorter than kmer lenth
  DT <- DT[nchar(kmer) >= k, , ]

  # transpose so can do function on columns
  DT <- transpose(DT)

  # For each CDR3 sequence, identify kmers and associated counts
  Pos_DT <- DT[, list(lapply(.SD, FUN = split_cdr3_into_positional_kmers, k = k, sample_names = sample.names, pos = c("start", "middle", "end"))), ]

  # Merge, sum, sort and write
  Pos_DT <- Pos_DT$V1 %>% bind_rows(.)
  Pos_DT <- Pos_DT[, lapply(.SD, FUN = sum), by = kmer]
  Pos_DT <- Pos_DT[order(kmer), , ]
  fwrite(Pos_DT, OutputFile) # Expects Output file to be csv.gz
  return(Pos_DT)
}
