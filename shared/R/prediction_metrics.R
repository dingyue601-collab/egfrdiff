# Weighted binary classification metrics
clamp_probability <- function(p) pmin(pmax(p,1e-8),1-1e-8)
safe_ratio <- function(x,d) ifelse(d>0,x/d,NA_real_)
prediction_metrics <- function(y,p,w,thresholds){
  stopifnot(length(y)==length(p),nrow(w)==length(y),all(is.finite(p)),all(p>=0&p<=1),
    all(y%in%c(0,1)),all(is.finite(w)),all(w>=0))
  ord<-order(p);ties<-cumsum(c(TRUE,diff(p[ord])!=0))
  pw<-rowsum(w[ord,,drop=FALSE]*y[ord],ties,reorder=FALSE)
  nw<-rowsum(w[ord,,drop=FALSE]*(1-y[ord]),ties,reorder=FALSE)
  n1<-colSums(pw);n0<-colSums(nw);nn<-n1+n0
  auc<-colSums(pw*(matrixStats::colCumsums(nw)-nw/2))/(n1*n0)
  revix<-rev(seq_len(nrow(pw)))
  cp<-matrixStats::colCumsums(pw[revix,,drop=FALSE]);cn<-matrixStats::colCumsums(nw[revix,,drop=FALSE])
  precision<-cp/(cp+cn);precision[!is.finite(precision)]<-0
  q<-clamp_probability(p)
  core<-rbind(AUC=auc,AP=colSums(pw[revix,,drop=FALSE]*precision)/n1,
    Brier=colSums(w*(p-y)^2)/nn,
    LogLoss=-colSums(w*(y*log(q)+(1-y)*log1p(-q)))/nn)
  lapply(thresholds,function(t){
    tp<-colSums(w*((p>=t)*y));fp<-colSums(w*((p>=t)*(1-y)));fn<-n1-tp;tn<-n0-fp
    den<-sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    rbind(core,Sensitivity=tp/n1,Specificity=tn/n0,PPV=safe_ratio(tp,tp+fp),NPV=safe_ratio(tn,tn+fn),
      Accuracy=(tp+tn)/nn,F1=safe_ratio(2*tp,2*tp+fp+fn),MCC=safe_ratio(tp*tn-fp*fn,den),
      TestingFraction=(tp+fp)/nn,MeanPrediction=colSums(w*p)/nn,Observed=n1/nn,
      TP=tp,FP=fp,FN=fn,TN=tn)
  })
}
