# Problem Set 04 starter code
#
# Please make sure your code runs on R version 3.2.2
# Due date: 2016-03-04 13:00

#' Support vector machine - Dual problem
#'
#' SVM classification for a numeric test matrix. The
#'   returned result is the vector of coefficients from
#'   the support vector machine (beta, *not* alpha!).
#'
#' @param X     an n by p numeric matrix; the data
#'                matrix of predictors
#' @param y     a length n numberic vector; the observed
#'                response
#' @param C     positive numeric value giving the cost of
#'                parameter in the support vector machine
#'
#' @return      a length p vector giving the coefficients of
#'                beta in the SVM
dualFunc <- function(alphas, Xs, ys) {
  print(class(alphas))
  res = -sum(alphas)
  for (i in 1:nrow(Xs)) {
    for (j in 1:nrow(Xs)){
      res = res + 1/ 2 * alphas[i] * alphas[j] * y[i] *y[j] * ((Xs[i] %*% Xs[j])[1,1])
    }}
  res
}
my_dual_svm <- function(X, y, C=1L) {
  alphas = c(rep(C/2,nrow(X))) # this is the right dimension for the output
  alphas = optim(par = alphas, fn = dualFunc, Xs = X, ys = y, method="L-BFGS-B", 
                 lower=rep(0, nrow(X)), upper=rep(C, nrow(X)))
  alphas
}

#' Support vector machine - Primal problem
#'
#' SVM classification for a numeric test matrix. The
#'   returned result is the vector of coefficients from
#'   the support vector machine (beta, *not* alpha!).
#'   The algorithm solves the primal problem, written
#'   as a penalized hinge loss, using the Pegasos algorithm
#'
#' @param X       an n by p numeric matrix; the data
#'                  matrix of predictors
#' @param y       a length n numberic vector; the observed
#'                  response
#' @param lam     positive numeric value giving the tuning
#'                  parameter in the (primal, penalized format)
#'                  of the support vector machine
#' @param k       positive integer giving the number of samples
#'                  selected in each iteration of the algorithm
#' @param T       positive integer giving the total number of
#'                   iterations to run
#'
#' @return      a length p vector giving the coefficients of
#'                beta in the SVM
my_primal_svm <- function(X, y, lam=1, k=5, T=100) {
  rep(0,ncol(X)) # this is the right dimension for the output
}



