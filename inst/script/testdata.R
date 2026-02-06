#' This code is used for generating the small test dataset for vignette and
#' testing.

library(SpatialExperimentIO)

dir = "/mnt/beegfs/mccarthy/backed_up/general/rlyu/Dataset/LFST_2022/GEO_2025/VUHD116A/relabel_output-XETG00048__0003817__VUHD116A__20230308__003730/outs/"
spe <- readXeniumSXE(dir, returnType = "SPE")
example_small <- spe[,sample(ncol(spe), 300)]
saveRDS(example_small, "inst/extdata/example_spe.rds")
