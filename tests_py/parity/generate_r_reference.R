#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[[1]] else file.path("tests_py", "parity", "reference")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(flexmix)
  library(hexbin)
  library(pbapply)
  library(Matrix)
  library(dbscan)
})

if (!exists("rowSums2", mode = "function")) {
  rowSums2 <- function(x, ...) {
    rowSums(as.matrix(x), ...)
  }
}

source(file.path("R", "pmm_model.R"))
source(file.path("R", "neighbour_offset.R"))
source(file.path("R", "denoist.R"))

test_mat <- readRDS(file.path("tests", "testthat", "testdata", "test_mat.rds"))
test_coords <- readRDS(file.path("tests", "testthat", "testdata", "test_coords.rds"))
test_tx <- readRDS(file.path("tests", "testthat", "testdata", "test_tx.rds"))

n_inits <- c(0.1, 0.2, 0.3)
distance <- 50
nbins <- 200
posterior_cutoff <- 0.6

off_fast <- local_offset_distance_with_background_fast(
  mat = test_mat,
  coords = test_coords,
  tx = test_tx,
  distance = distance,
  nbins = nbins,
  cl = 1,
  verbose = FALSE
)

res_fast <- denoist(
  mat = test_mat,
  coords = test_coords,
  tx = test_tx,
  distance = distance,
  nbins = nbins,
  posterior_cutoff = posterior_cutoff,
  n_inits = n_inits,
  cl = 1,
  neighbour_mode = "fast",
  out_dir = NULL,
  verbose = FALSE
)

write.csv(
  data.frame(gene = rownames(test_mat)),
  file = file.path(out_dir, "gene_names.csv"),
  row.names = FALSE
)
write.csv(
  data.frame(cell = colnames(test_mat)),
  file = file.path(out_dir, "cell_names.csv"),
  row.names = FALSE
)
write.csv(
  as.matrix(test_mat),
  file = file.path(out_dir, "raw_counts.csv"),
  row.names = TRUE
)
write.csv(
  as.data.frame(test_coords),
  file = file.path(out_dir, "coords.csv"),
  row.names = TRUE
)
write.csv(
  as.data.frame(test_tx),
  file = file.path(out_dir, "transcripts.csv"),
  row.names = FALSE
)
write.csv(
  as.matrix(off_fast),
  file = file.path(out_dir, "offset_fast.csv"),
  row.names = TRUE
)
write.csv(
  as.matrix(res_fast$memberships),
  file = file.path(out_dir, "memberships_fast.csv"),
  row.names = TRUE
)
write.csv(
  as.matrix(res_fast$adjusted_counts),
  file = file.path(out_dir, "adjusted_counts_fast.csv"),
  row.names = TRUE
)

params <- do.call(rbind, lapply(seq_along(res_fast$params), function(i) {
  p <- res_fast$params[[i]]
  data.frame(
    cell_index = i,
    lambda1 = p$lambda1,
    lambda2 = p$lambda2,
    pi = p$pi,
    log_lik = p$log_lik
  )
}))
write.csv(params, file = file.path(out_dir, "params_fast.csv"), row.names = FALSE)

metadata <- data.frame(
  metric = c("distance", "nbins", "posterior_cutoff", "n_inits"),
  value = c(distance, nbins, posterior_cutoff, paste(n_inits, collapse = ","))
)
write.csv(metadata, file = file.path(out_dir, "metadata.csv"), row.names = FALSE)

message("Wrote R parity reference files to: ", normalizePath(out_dir))
