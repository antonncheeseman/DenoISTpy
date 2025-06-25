utils::globalVariables(c("feature_name", "hexbin_id", "count"))


#' @importFrom pbapply pblapply
#' @importFrom stats dpois runif
#' @importFrom SpatialExperiment spatialCoords
#' @importFrom SummarizedExperiment assay
#' @importFrom parallel makeCluster clusterExport parLapply stopCluster
#' @export
#' @title DenoIST
#' @description
#' DenoIST (Denoising Image-based Spatial Transcriptomics) is a method for
#' identifying and removing contamination artefacts in
#' image-based single-cell transcriptomics (IST) data.
#' It uses a transposed Poisson mixture model to identify contamination.
#' @param mat A matrix of counts (genes x cells), or a SpatialExperiment object.
#' @param coords A data frame of coordinates (n_cells x 2).
#' @param tx A data frame of transcript with x, y and qv columns.
#' @param tx_x Column name for the x coordinates in the transcripts dataframe. Default is 'x'.
#' @param tx_y Column name for the y coordinates in the transcripts dataframe. Default is 'y'.
#' @param feature_label Column name for the gene of each transcript in the transcripts dataframe. Default is 'gene'.
#' @param distance The maximum distance to consider for local background estimation.
#' @param nbins The number of bins to use for hexagonal binning.
#' @param posterior_cutoff The cutoff for posterior probability to determine contamination.
#' @param cl The number of cores to use for parallel processing.
#' @param out_dir The output directory to save the results.
#' @return A list containing the following elements:
#' \item{memberships}{A matrix of memberships for each gene in each cell.}
#' \item{adjusted_counts}{A matrix of adjusted counts for each gene in each cell.}
#' \item{params}{A list of parameters for each gene.}
#' @details
#' The function calculates local background using hexagonal binning and applies
#' a Poisson mixture model to identify contamination.
#' It returns a matrix of memberships and adjusted counts for each gene in each cell.
#' @examples
#' # Load example data
#' set.seed(42)
#' mat <- matrix(rpois(1000, lambda = 10), nrow = 10, ncol = 100)
#' rownames(mat) <- paste0("gene", 1:10)
#' coords <- data.frame(x = rnorm(100), y = rnorm(100))
#' tx <- data.frame(x = c(rnorm(500), rnorm(500, 3)),
#'                  y = c(rnorm(500), rnorm(500, 3)),
#'                  qv = rep(30, 1000), gene = paste0('gene', 1:10))
#' # Run DenoIST
#' result <- denoist(mat, tx, coords, distance = 1, nbins = 50, cl = 1, out_dir = NULL)
#' # Check results
#' print(result$memberships[1:5, 1:5])
#' print(result$adjusted_counts[1:5, 1:5])
#' print(result$params[[1]])
denoist <- function(mat, tx, coords = NULL,
                    tx_x = "x",
                    tx_y = "y",
                    feature_label = "gene",
                    distance = 50,
                    nbins = 200,
                    posterior_cutoff = 0.6,
                    cl = 1,
                    out_dir = NULL){
  # TODO:check input type
  if(inherits(mat, "SpatialExperiment")){
    coords <- spatialCoords(mat)
    mat <- assay(mat)
    #remove NegControl and BLANKS
    mat <- mat[!grepl("NegControl|BLANK|Unassigned", rownames(mat)),]
  }else if(is.null(coords)){
    stop("coords must be provided")
  }

  #print(coords[1:5,1:2])
  #print(colnames(coords))
  #print(mat[1:5, 1:5])
  #print(tx[1:5,])
  # calculate neighbour_offset
  message("Calculating neighbour offset...")
  off_mat <- local_offset_distance_with_background(mat = mat,
                                                   coords = coords,
                                                   tx = tx,
                                                   tx_x = tx_x,
                                                   tx_y = tx_y,
                                                   feature_label = feature_label,
                                                   distance = distance,
                                                   nbins = nbins,
                                                   cl = cl)
  # Clear tx just in case to save mem
  rm(tx)
  gc()

  # Function to parallelize over
  apply_poisson_mixture_single <- function(mat, off_mat, posterior_cutoff){
    function(idx) { # friendship ended with OOP, FP is now my best friend
      c_vec <- mat[,idx]
      s_vec <- off_mat[,idx]
      result <- tryCatch({
        out <- solve_poisson_mixture(c_vec, s_vec, posterior_cutoff = posterior_cutoff)
        return(out)
      }, error = function(e) {
        out <- list(memberships = rep(1, length(c_vec)),
                    posterior = rep(1, length(c_vec)),
                    lambda1 = NA, lambda2 = NA, pi = NA)
        return(out)
      })
      return(result)
    }
  }

  if(cl > 1){
    message("Setting up parallel processing...")
    clust <- makeCluster(cl-1, type = "FORK")
    message("Exporting variables to workers...")
    clusterExport(clust, c("mat", "off_mat", "solve_poisson_mixture", "posterior_cutoff"), envir = environment())

    # Apply the Poisson mixture model
    message("Applying the Poisson mixture model...")
    results <- parLapply(cl = clust, 1:ncol(mat), apply_poisson_mixture_single(mat, off_mat, posterior_cutoff))
    stopCluster(clust)
  }else{
    # use pblapply normally without multiple processing
    message("Applying the Poisson mixture model without parallel processing...")
    results <- pblapply(1:ncol(mat), apply_poisson_mixture_single(mat, off_mat, posterior_cutoff), cl = cl)
  }


  # return neighbour_offset, adjusted_counts, posterior, params
  message("Tidying up results...")
  memberships_matrix <- do.call(cbind, lapply(results, function(res) res["memberships"]))
  memberships_matrix <- do.call(cbind, memberships_matrix)
  colnames(memberships_matrix) <- colnames(mat)
  rownames(memberships_matrix) <- rownames(mat)

  adjusted_counts <- mat * memberships_matrix
  colnames(adjusted_counts) <- colnames(mat)
  rownames(adjusted_counts) <- rownames(mat)

  # save the results
  if(!is.null(out_dir)){
    if(!dir.exists(out_dir)){
      dir.create(out_dir, recursive = TRUE)
    }
    saveRDS(list(memberships = memberships_matrix,
                 adjusted_counts = adjusted_counts,
                 params = results),
            file = paste0(out_dir, "/denoist_results.rds"))
  }

  return(list(memberships = memberships_matrix,
              adjusted_counts = adjusted_counts,
              params = results))
}
