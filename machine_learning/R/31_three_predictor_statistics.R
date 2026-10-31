options(encoding = "UTF-8", stringsAsFactors = FALSE)
source("shared/R/config.R")
suppressPackageStartupMessages({library(data.table); library(jsonlite); library(pROC)})
setDTthreads(1L)
set.seed(123)
base <- "results"
out <- file.path(base, "clinical_reference_tabm3")
prep <- readRDS(file.path(base, "ml_imputation_inputs.rds"))
rel <- readRDS(file.path(base, "metrics/Release_metrics.rds"))
ref <- list(train = rel$reference$TrainingOOF, evaluation = rel$reference$Validation)
y <- ref$evaluation$Label
cluster <- ref$evaluation$Cluster
p9 <- rel$probabilities$TabM9$Validation
t9 <- rel$parameters[Algorithm == "TabM9", Threshold80]
stopifnot(length(y) == 3038L, sum(y) == 658L, length(p9) == length(y),
          identical(ref$train$Row, seq_len(nrow(prep$train))),
          identical(ref$evaluation$Row, seq_len(nrow(prep$evaluation_raw))),
          identical(ref$train$Label, prep$y), identical(ref$train$Fold, prep$foldid),
          identical(ref$train$Cluster, match(prep$train_keys$PatientID, unique(prep$train_keys$PatientID))),
          identical(cluster, match(prep$evaluation_keys$PatientID, unique(prep$evaluation_keys$PatientID))),
          !length(intersect(prep$train_keys$PatientID, prep$evaluation_keys$PatientID)))

metrics <- function(y, p, selected) {
  n1 <- sum(y); n0 <- sum(1-y)
  tp <- sum(selected & y == 1); fp <- sum(selected & y == 0)
  fn <- n1-tp; tn <- n0-fp
  c(AUC = (sum(rank(p)[y == 1])-n1*(n1+1)/2)/(n1*n0),
    Brier = mean((p-y)^2), Accuracy = (tp+tn)/length(y),
    Precision = tp/(tp+fp), Sensitivity = tp/n1, Specificity = tn/n0,
    NPV = tn/(tn+fn), F1 = 2*tp/(2*tp+fp+fn), TestingFraction = mean(selected),
    TP = tp, FP = fp, FN = fn, TN = tn)
}
calibration <- function(y, p) {
  z <- qlogis(pmin(pmax(p, 1e-8), 1-1e-8))
  c(CalibrationIntercept = unname(coef(glm(y ~ 1, offset = z, family = binomial()))),
    CalibrationSlope = unname(coef(glm(y ~ z, family = binomial()))[2]))
}
groups <- split(seq_along(y), cluster)
B <- 5000L
ci_names <- c("AUC", "Brier", "Accuracy", "Precision", "Sensitivity", "Specificity", "NPV", "F1", "TestingFraction")

  old <- readRDS(file.path(base, "clinical_reference/Clinical_reference_results.rds"))
  raw <- rowMeans(fread(file.path(out,"TabM3_OOF.csv")))
  rawv <- rowMeans(fread(file.path(out,"TabM3_Validation.csv")))
  yt <- ref$train$Label
  stopifnot(length(raw)==6852L, length(rawv)==3038L)
  cal <- glm(yt ~ qlogis(pmin(pmax(raw,1e-8),1-1e-8)), family=binomial())
  ab <- unname(coef(cal)); stopifnot(ab[2]>0)
  map <- function(p) plogis(ab[1]+ab[2]*qlogis(pmin(pmax(p,1e-8),1-1e-8)))
  pt <- map(raw); pv <- map(rawv)
  threshold3 <- sort(pt[yt==1],decreasing=TRUE)[ceiling(.8*sum(yt))]
  ps <- list(TabM9=p9, TabM3=pv, Logistic3=old$probabilities$Logistic3)
  ts <- c(TabM9=t9, TabM3=threshold3, Logistic3=old$thresholds["Logistic3"])
  names(ts) <- names(ps)
  capacity <- function(p, selected) {
    z <- rep(FALSE,length(p)); z[order(-p, seq_along(p))[seq_len(sum(selected))]] <- TRUE; z
  }
  select <- function(name, rule, ix) {
    if(rule=="Training80" || name=="TabM9") ps[[name]][ix]>=ts[name]
    else capacity(ps[[name]][ix],p9[ix]>=t9)
  }
  rows <- CJ(Rule=c("Training80","Matched_testing"), Model=names(ps))
  performance <- rbindlist(lapply(seq_len(nrow(rows)),function(i) {
    n<-rows$Model[i]; r<-rows$Rule[i]
    as.data.table(c(list(Rule=r,Model=n,N=length(y),Patients=uniqueN(cluster),Threshold=if(r=="Training80"||n=="TabM9") ts[n] else NA_real_),
      as.list(c(metrics(y,ps[[n]],select(n,r,seq_along(y))),calibration(y,ps[[n]])))))
  }))
  boot <- array(NA_real_,c(B,nrow(rows),length(ci_names)))
  for(b in seq_len(B)) {
    ix<-unlist(groups[sample.int(length(groups),length(groups),replace=TRUE)],use.names=FALSE)
    for(i in seq_len(nrow(rows))) boot[b,i,]<-metrics(y[ix],ps[[rows$Model[i]]][ix],select(rows$Model[i],rows$Rule[i],ix))[ci_names]
    if(b%%1000L==0L) {cat("BOOTSTRAP",b,"\n");flush.console()}
  }
  intervals<-rbindlist(lapply(seq_len(nrow(rows)),function(i) rbindlist(lapply(seq_along(ci_names),function(j) {
    q<-quantile(boot[,i,j],c(.025,.975),na.rm=TRUE)
    data.table(Rule=rows$Rule[i],Model=rows$Model[i],Metric=ci_names[j],Estimate=performance[[ci_names[j]]][i],Lower95=q[1],Upper95=q[2])
  }))))
  differences<-rbindlist(lapply(c("Training80","Matched_testing"),function(r) rbindlist(lapply(c("TabM3","Logistic3"),function(n) {
    a<-which(rows$Rule==r & rows$Model=="TabM9"); c<-which(rows$Rule==r & rows$Model==n)
    rbindlist(lapply(seq_along(ci_names),function(j) {
      q<-quantile(boot[,a,j]-boot[,c,j],c(.025,.975),na.rm=TRUE)
      data.table(Rule=r,Comparator=n,Metric=ci_names[j],Difference=performance[[ci_names[j]]][a]-performance[[ci_names[j]]][c],Lower95=q[1],Upper95=q[2])
    }))
  }))))
  stopifnot(abs(as.numeric(pROC::auc(y,pv,quiet=TRUE,direction="<"))-performance[Model=="TabM3",AUC][1])<1e-10)
  result<-list(performance=performance,intervals=intervals,differences=differences,
    training=metrics(yt,pt,pt>=threshold3),platt=ab,threshold80=threshold3,bootstrap=B,seed=123,
    probabilities=list(Training=pt,Validation=pv),configuration="Same fixed TabM architecture, weighting and 42-epoch schedule; no three-input hyperparameter search")
  saveRDS(result,file.path(out,"TabM3_reference_results.rds"))
  write_json(result[setdiff(names(result),"probabilities")],file.path(out,"TabM3_reference_results.json"),pretty=TRUE,auto_unbox=TRUE,digits=16)
  fwrite(performance,file.path(out,"Reference_performance.csv"));fwrite(intervals,file.path(out,"Reference_intervals.csv"))
  fwrite(differences,file.path(out,"Reference_paired_differences.csv"))
  print(performance)
cat("THREE_PREDICTOR_COMPARISON_COMPLETE\n")
