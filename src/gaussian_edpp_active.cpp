
#include "utilities.h"
#include <vector>

void Free_memo_edpp(double *a, double *r, int *discard_beta, double *theta, double *v1, double *v2, double *o);

// V2 - <v1, v2> / ||v1||^2_2 * V1
void update_pv2(double *pv2, double *v1, double *v2, int n);

// apply EDPP while keeping track of those leaving just now
void edpp_screen_with_news(int *discard_beta, XPtr<BigMatrix> xpMat, double *o, 
                 int *row_idx, vector<int> &col_idx,
                 NumericVector &center, NumericVector &scale, int n, int p, 
                 double rhs, bool *newly_entered) {
  MatrixAccessor<double> xAcc(*xpMat);
  
  int j, jj;
  double lhs, sum_xy, sum_y;
  double *xCol;
  
#pragma omp parallel for private(j, lhs, sum_xy, sum_y) default(shared) schedule(static) 
  for (j = 0; j < p; j++) {
    sum_xy = 0.0;
    sum_y = 0.0;
    
    jj = col_idx[j];
    xCol = xAcc[jj];
    for (int i=0; i < n; i++) {
      sum_xy = sum_xy + xCol[row_idx[i]] * o[i];
      sum_y = sum_y + o[i];
    }
    lhs = fabs((sum_xy - center[jj] * sum_y) / scale[jj]);
    if (lhs < rhs) {
      discard_beta[j] = 1;
    } else {     
      if (discard_beta[j]) newly_entered[j] = true;
      discard_beta[j] = 0;
    }
  }
}

// check edpp set
int check_edpp_set(int *ever_active, int *discard_beta, vector<double> &z, 
                   XPtr<BigMatrix> xpMat, int *row_idx, vector<int> &col_idx,
                   NumericVector &center, NumericVector &scale, double *a,
                   double lambda, double sumResid, double alpha, 
                   double *r, double *m, int n, int p) {
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol, sum, l1, l2;
  int j, jj, violations = 0;
  
#pragma omp parallel for private(j, sum, l1, l2) reduction(+:violations) schedule(static) 
  for (j = 0; j < p; j++) {
    if (ever_active[j] == 0 && discard_beta[j] == 0) {
      jj = col_idx[j];
      xCol = xAcc[jj];
      sum = 0.0;
      for (int i=0; i < n; i++) {
        sum = sum + xCol[row_idx[i]] * r[i];
      }
      z[j] = (sum - center[jj] * sumResid) / (scale[jj] * n);
      l1 = lambda * m[jj] * alpha;
      l2 = lambda * m[jj] * (1 - alpha);
      if (fabs(z[j] - a[j] * l2) > l1) {
        ever_active[j] = 1;
        violations++;
      }
    }
  }
  return violations;
}

// check edpp set
int check_edpp_set(int *ever_active, int *discard_beta, vector<double> &z, 
                   XPtr<BigMatrix> xpMat, int *row_idx, vector<int> &col_idx,
                   NumericVector &center, NumericVector &scale, double *a,
                   double lambda, double sumResid, double alpha, 
                   double *r, double *m, int n, int p,
                   int &steps, int &stepsum, double *r_diff, double *sum_prev, double *var,
                   bool *newly_entered) {
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol, sum, sqr_sum, l1, l2;
  int j, jj, violations = 0;
  
  int nsample = n / 10;
  
  #pragma omp parallel for private(j, sum, l1, l2) reduction(+:violations,steps,stepsum) schedule(static) 
  
  for (j = 0; j < p; j++) {
    if (ever_active[j] == 0 && discard_beta[j] == 0) {
      jj = col_idx[j];
      xCol = xAcc[jj];
      
      l1 = lambda * m[jj] * alpha;
      l2 = lambda * m[jj] * (1 - alpha);
      
      sum = 0.0;
      sqr_sum = 0.0;
      for (int i=0; i < nsample; i++) {
        double current_sample = xCol[row_idx[i]] * r_diff[i];
        sum = sum + current_sample;
        sqr_sum = sqr_sum + current_sample * current_sample;
      }
      
      double variance = sqr_sum / nsample - sum / nsample * sum / nsample;
      
      var[j] += variance / nsample;
      sum_prev[j] -= sum * n / nsample;
      
      z[j] = (sum_prev[j] - center[jj] * sumResid) / (scale[jj] * n);
      
      if (newly_entered[j] || is_hypothesis_accepted(l1,  (z[j]-a[j] * l2), sqrt(var[j])/scale[jj] ,0.01)) {
        stepsum += n;
        steps++;
        sum = 0.0;
        for (int i=0; i < n; i++) {
          sum = sum + xCol[row_idx[i]] * r[i];
        }
        sum_prev[j] = sum;        
        var[j] = 0;
        
        z[j] = (sum - center[jj] * sumResid) / (scale[jj] * n);
        if (fabs(z[j] - a[j] * l2) > l1) {
          ever_active[j] = 1;
          violations++;
        }
        newly_entered[j] = false;
      }  
      else {
        steps++;
        stepsum += nsample;
      }      
    }
  }
  // zeroing out r_diff
  for (j = 0; j < n; j++) {
    r_diff[j] = 0;
  }
  
  return violations;
}

// Coordinate descent for gaussian models
RcppExport SEXP cdfit_gaussian_edpp_active(SEXP X_, SEXP y_, SEXP row_idx_, SEXP lambda_, 
                                    SEXP nlambda_, SEXP lam_scale_,
                                    SEXP lambda_min_, SEXP alpha_, 
                                    SEXP user_, SEXP eps_, SEXP max_iter_, 
                                    SEXP multiplier_, SEXP dfmax_, SEXP ncore_) {
  XPtr<BigMatrix> xMat(X_);
  double *y = REAL(y_);
  int *row_idx = INTEGER(row_idx_);
  double lambda_min = REAL(lambda_min_)[0];
  double alpha = REAL(alpha_)[0];
  int n = Rf_length(row_idx_); // number of observations used for fitting model
  int p = xMat->ncol();
  int lam_scale = INTEGER(lam_scale_)[0];
  int L = INTEGER(nlambda_)[0];
  int user = INTEGER(user_)[0];
  double eps = REAL(eps_)[0];
  int max_iter = INTEGER(max_iter_)[0];
  double *m = REAL(multiplier_);
  int dfmax = INTEGER(dfmax_)[0];
  
  NumericVector lambda(L);
  NumericVector center(p);
  NumericVector scale(p);
  int p_keep = 0;
  int *p_keep_ptr = &p_keep;
  
  vector<int> col_idx;
  vector<double> z;
  double lambda_max = 0.0;
  double *lambda_max_ptr = &lambda_max;
  int xmax_idx = 0;
  int *xmax_ptr = &xmax_idx;
  
  // set up omp
  int useCores = INTEGER(ncore_)[0];
#ifdef BIGLASSO_OMP_H_
  int haveCores = omp_get_num_procs();
  if(useCores < 1) {
    useCores = haveCores;
  }
  omp_set_dynamic(0);
  omp_set_num_threads(useCores);
#endif
  
  standardize_and_get_residual(center, scale, p_keep_ptr, col_idx, z, 
                               lambda_max_ptr, xmax_ptr, xMat, 
                               y, row_idx, lambda_min, alpha, n, p);
  p = p_keep; // set p = p_keep, only loop over columns whose scale > 1e-6

  // Objects to be returned to R
  arma::sp_mat beta = arma::sp_mat(p, L); //Beta
  double *a = Calloc(p, double); //Beta from previous iteration
  NumericVector loss(L);
  IntegerVector iter(L);
  IntegerVector n_reject(L);
 
  double l1, l2, shift;
  double max_update, update, thresh; // for convergence check
  int i, j, jj, l, violations, lstart; //temp index
  int *ever_active = Calloc(p, int); // ever-active set
  int *discard_beta = Calloc(p, int); // index set of discarded features;
  double *r = Calloc(n, double);
  double *r_diff = Calloc(n, double);
  
  for (i = 0; i < n; i++) r[i] = y[i];
  for (i = 0; i < n; i++) r_diff[i] = -y[i];
  
  // hypothesis testing, differencing
  double *sum_prev = Calloc(p, double);
  double *var = Calloc(p, double);
  bool *newly_entered = Calloc(p, bool);
  
  int steps = 0, stepsum = 0;
  
  double sumResid = sum(r, n);
  loss[0] = gLoss(r, n);
  thresh = eps * loss[0] / n;
  
  // EDPP
  double *theta = Calloc(n, double);
  double *v1 = Calloc(n, double);
  double *v2 = Calloc(n, double);
  double *pv2 = Calloc(n, double);
  double *o = Calloc(n, double);
  double pv2_norm = 0;
  
  // lambda, equally spaced on log scale
  if (user == 0) {
    if (lam_scale) {
      // set up lambda, equally spaced on log scale
      double log_lambda_max = log(lambda_max);
      double log_lambda_min = log(lambda_min*lambda_max);
      
      double delta = (log_lambda_max - log_lambda_min) / (L-1);
      for (l = 0; l < L; l++) {
        lambda[l] = exp(log_lambda_max - l * delta);
      }
    } else { // equally spaced on linear scale
      double delta = (lambda_max - lambda_min*lambda_max) / (L-1);
      for (l = 0; l < L; l++) {
        lambda[l] = lambda_max - l * delta;
      }
    }
    lstart = 1;
    n_reject[0] = p;
  } else {
    lstart = 0;
    lambda = Rcpp::as<NumericVector>(lambda_);
  } 
  
  // compute v1 for lambda_max
  double xty = crossprod_bm(xMat, y, row_idx, center[xmax_idx], scale[xmax_idx], n, xmax_idx);
  
  // Path
  for (l = lstart; l < L; l++) {
    if (l != 0 ) {
      int nv = 0;
      for (int j=0; j<p; j++) {
        if (a[j] != 0) nv++;
      }
      if (nv > dfmax) {
        for (int ll=l; ll<L; ll++) iter[ll] = NA_INTEGER;
        Free(ever_active);
        Free_memo_edpp(a, r, discard_beta, theta, v1, v2, o);
        return List::create(beta, center, scale, lambda, loss, iter,  n_reject, Rcpp::wrap(col_idx));
      }
      // update theta and v1
      for (i = 0; i < n; i++) {
        theta[i] = r[i] / lambda[l-1];
        if (lambda[l-1] < lambda_max) {
          v1[i] = y[i] / lambda[l-1] - theta[i];
        } else {
          v1[i] = sign(xty) * get_elem_bm(xMat, center[xmax_idx], scale[xmax_idx], row_idx[i], xmax_idx);        
        }
      }
    } else { // lam[0]
      for (i = 0; i < n; i++) {
        theta[i] = r[i] / lambda_max;
        if (lambda[l] < lambda_max) {
          v1[i] = y[i] / lambda[l] - theta[i];
        } else {
          v1[i] = sign(xty) * get_elem_bm(xMat, center[xmax_idx], scale[xmax_idx], row_idx[i], xmax_idx);        
        }
      }
    } 
    // update v2:
    for (i = 0; i < n; i++) {
      v2[i] = y[i] / lambda[l] - theta[i];
    }
    //update pv2:
    update_pv2(pv2, v1, v2, n);
    // update norm of pv2;
    for (i = 0; i < n; i++) {
      pv2_norm += pow(pv2[i], 2);
    }
    pv2_norm = pow(pv2_norm, 0.5);
    // update o
    for (i = 0; i < n; i++) {
      o[i] = theta[i] + 0.5 * pv2[i];
    }
    double rhs = n - 0.5 * pv2_norm * sqrt(n); 
    
    // apply EDPP
    edpp_screen_with_news(discard_beta, xMat, o, row_idx, col_idx, center, scale, n, p, rhs, newly_entered);
    n_reject[l] = sum(discard_beta, p);
    
    while (iter[l] < max_iter) {
      while (iter[l] < max_iter) {
        iter[l]++;
        
        max_update = 0.0;
        for (j = 0; j < p; j++) {
          if (ever_active[j]) {
            jj = col_idx[j];
            z[j] = crossprod_resid(xMat, r, sumResid, row_idx, center[jj], scale[jj], n, jj) / n + a[j];
            l1 = lambda[l] * m[jj] * alpha;
            l2 = lambda[l] * m[jj] * (1-alpha);
            beta(j, l) = lasso(z[j], l1, l2, 1);
            
            shift = beta(j, l) - a[j];
            if (shift != 0) {
              // compute objective update for checking convergence
              //update =  z[j] * shift - 0.5 * (1 + l2) * (pow(beta(j, l+1), 2) - pow(a[j], 2)) - l1 * (fabs(beta(j, l+1)) -  fabs(a[j]));
              update = pow(beta(j, l) - a[j], 2);
              if (update > max_update) {
                max_update = update;
              }
              update_resid_diff(xMat, r, shift, row_idx, center[jj], scale[jj], n, jj, r_diff);
              sumResid = sum(r, n); //update sum of residual
              a[j] = beta(j, l); //update a
            }
            // update ever active sets
            if (beta(j, l) != 0) {
              ever_active[j] = 1;
            } 
          }
        }
        // Check for convergence
        if (max_update < thresh) break;
      }
    
      // Scan for violations in edpp set
      violations = check_edpp_set(ever_active, discard_beta, z, xMat, row_idx, col_idx, center, scale, a, lambda[l], sumResid, alpha, r, m, n, p,
                                  steps, stepsum, r_diff, sum_prev, var, newly_entered); 
      if (violations == 0) {
        loss[l] = gLoss(r, n);
        break;
      }
    }
  }
  Rprintf("\n Avg steps: %f %d %d\n", ((double)stepsum)/steps, steps,stepsum);
  
  Free(ever_active);
  Free_memo_edpp(a, r, discard_beta, theta, v1, v2, o);
  return List::create(beta, center, scale, lambda, loss, iter, n_reject, Rcpp::wrap(col_idx));
}