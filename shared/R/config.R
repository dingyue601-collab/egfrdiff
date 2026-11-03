# Paths and input readers
options(stringsAsFactors = FALSE, encoding = "UTF-8")
if (.Platform$OS.type == "windows") invisible(Sys.setlocale("LC_CTYPE", "English_United States.utf8"))
prepared_dir <- Sys.getenv("EGFR_INPUT_DIR", "data/prepared")
reference_dir <- Sys.getenv("EGFR_REFERENCE_DIR", "data/reference")
input_file <- function(path) {
  if (file.exists(path)) return(path)
  relative <- sub("^.*?/results/", "", gsub("\\\\", "/", path))
  relative <- sub("^results/", "", relative)
  choices <- c(file.path(prepared_dir, relative), file.path(prepared_dir, basename(path)), file.path(reference_dir, relative), file.path(reference_dir, basename(path)))
  found <- choices[file.exists(choices)]
  if (!length(found)) stop("Input not found: ", basename(path))
  found[1L]
}
readRDS <- function(file, ...) base::readRDS(input_file(file), ...)
fread <- function(file, ...) data.table::fread(input_file(file), ...)
read_json <- function(path, ...) jsonlite::read_json(input_file(path), ...)
fromJSON <- function(txt, ...) jsonlite::fromJSON(input_file(txt), ...)
readLines <- function(con, ...) base::readLines(input_file(con), ...)
dir.create("results", showWarnings = FALSE)
dir.create("results/publication", showWarnings = FALSE)
