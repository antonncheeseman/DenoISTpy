test_mat <- readRDS(test_path("testdata", "test_mat.rds"))
test_coords <- readRDS(test_path("testdata", "test_coords.rds"))
test_tx <- readRDS(test_path("testdata", "test_tx.rds"))
test_spe <- readRDS(test_path("testdata", "test_spe.rds"))

test_that("DenoIST works with SpatialExperiment input", {
   # Test the function with the provided data
  res <- denoist(mat = test_spe,
              coords = NULL,
              tx = test_tx,
              distance = 50, nbins = 200, cl = 1,
              out_dir = "denoist_results", verbose = TRUE)
  expect_length(res, 3)
})


test_that("DenoIST works with Matrix input", {
  # Test the function with the provided data
  res <- denoist(mat = test_mat,
               coords = test_coords,
               tx = test_tx,
               distance = 50, nbins = 200, cl = 1, verbose = TRUE)
  expect_length(res, 3)
})

test_that("DenoIST can be sped up with less inits", {
  # Test the function with the provided data
  res <- denoist(mat = test_mat,
                 coords = test_coords,
                 tx = test_tx,
                 distance = 50, nbins = 200, cl = 1, n_inits = 5, verbose = TRUE)
  expect_length(res, 3)
})

test_that("DenoIST takes vector as init input", {
  i <- c(0.1, 0.2, 0.3, 0.4, 0.5)
  # Test the function with the provided data
  res <- denoist(mat = test_mat,
                 coords = test_coords,
                 tx = test_tx,
                 distance = 50, nbins = 200, cl = 1, n_inits = i, verbose = TRUE)
  expect_length(res, 3)
})
