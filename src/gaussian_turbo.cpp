#include "utilities.h"
#include <vector>

void Free_memo(double *a, double *r, int *e1);



// check KKT conditions over features in the rest set
int check_inactive_set(int *e1, vector<double> &z, XPtr<BigMatrix> xpMat, int *row_idx, 
                       vector<int> &col_idx, NumericVector &center, NumericVector &scale, double *a,
                       double lambda, double sumResid, double alpha, double *r, double *m, int n, int p, int &steps, int &stepsum,
                       double *r_diff, double *sum_prev, double *var, int *start_pos) {
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol, sum, sqr_sum, l1, l2;
  int nsample = n / 10;
  int j, jj, violations = 0;
#pragma omp parallel for private(j, sum, sqr_sum, l1, l2) reduction(+:violations, steps, stepsum) schedule(static) 
  for (j = 0; j < p; j++) {
    if (e1[j] == 0) {
      jj = col_idx[j];
      xCol = xAcc[jj];
      sum = 0.0;
      sqr_sum = 0.0;
      int n_current_sample = max(50, n/1024);
      nsample = 0;      
      
      double current_scale = (scale[jj] * n);
      double variance;
      l1 = lambda * m[jj] * alpha;
      l2 = lambda * m[jj] * (1 - alpha);
      
      //do {        
        n_current_sample = n / 10; // n_current_sample * 2; 
        for (int i = start_pos[j]; i < start_pos[j] + n_current_sample; i++) {
          double current_sample = xCol[row_idx[i % n]] * r_diff[i % n];
          sum = sum + current_sample;
          sqr_sum = sqr_sum + current_sample * current_sample;
        }              
        nsample = nsample + n_current_sample;

        variance = sqr_sum / nsample - sum / nsample * sum / nsample;
        start_pos[j] = (start_pos[j] + n_current_sample) % n;       
        z[j] = ((sum_prev[j] + sum * n / nsample) - center[jj] * sumResid) / current_scale;
      //}  while (nsample<n/4 && is_hypothesis_accepted(l1,  (z[j]-a[j] * l2), sqrt(var[j] + variance / nsample)/scale[jj] ,0.0001));
      
      sum_prev[j] += sum * n / nsample;
      var[j] += variance / nsample;

  
      if (is_hypothesis_accepted(l1,  (z[j]-a[j] * l2), sqrt(var[j])/scale[jj] ,0.0001)) {
        stepsum += n;
        steps++;
        sum = 0.0;
        for (int i=0; i < n; i++) {
          sum = sum + xCol[row_idx[i]] * r[i];
        }
        sum_prev[j] = sum;
        z[j] = (sum - center[jj] * sumResid) / (scale[jj] * n);
        if (fabs(z[j] - a[j] * l2) > l1) {
          e1[j] = 1;
          violations++;
        }
        var[j] = 0;
      }        
      
      else {
        double true_sum = 0.0;
        for (int i=0; i < n; i++) {
          true_sum = true_sum + xCol[row_idx[i]] * r[i];
        }
        double true_z = (true_sum - center[jj] * sumResid) / (scale[jj] * n);
        if (fabs(true_z - a[j] * l2) > l1) {
          Rprintf("%d: %f %f %f var %.4e current %.4e scaled %.4e__ %d\n",j,true_z - a[j] * l2, z[j] - a[j] * l2, l1, var[j], variance, variance/nsample, start_pos[j]);
        }
        
        
        
        steps++;
        stepsum += nsample;
      }
    }
  }
  
  for (int i = 0; i < n ; i++) r_diff[i] = 0;
  
  return violations;
}

// Coordinate descent for gaussian models
RcppExport SEXP cdfit_gaussian_turbo(SEXP X_, SEXP y_, SEXP row_idx_, 
                               SEXP lambda_, SEXP nlambda_, 
                               SEXP lam_scale_, SEXP lambda_min_, 
                               SEXP alpha_, SEXP user_, SEXP eps_, 
                               SEXP max_iter_, SEXP multiplier_, SEXP dfmax_, 
                               SEXP ncore_, SEXP verbose_) {
  XPtr<BigMatrix> xMat(X_);
  double *y = REAL(y_);
  int *row_idx = INTEGER(row_idx_);
  double lambda_min = REAL(lambda_min_)[0];
  double alpha = REAL(alpha_)[0];
  int n = Rf_length(row_idx_); // number of observations used for fitting model
  
  int p = xMat->ncol();
  int L = INTEGER(nlambda_)[0];
  int lam_scale = INTEGER(lam_scale_)[0];
  int user = INTEGER(user_)[0];
  int verbose = INTEGER(verbose_)[0];
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
  
  if (verbose) {
    char buff1[100];
    time_t now1 = time (0);
    strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
    Rprintf("\nPreprocessing start: %s\n", buff1);
  }
  
  // standardize: get center, scale; get p_keep_ptr, col_idx; get z, lambda_max, xmax_idx;
  standardize_and_get_residual(center, scale, p_keep_ptr, col_idx, z, lambda_max_ptr, xmax_ptr, xMat, 
                               y, row_idx, lambda_min, alpha, n, p);
  
  p = p_keep;   // set p = p_keep, only loop over columns whose scale > 1e-6
  
  if (verbose) {
    char buff1[100];
    time_t now1 = time (0);
    strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
    Rprintf("Preprocessing end: %s\n", buff1);
    Rprintf("\n-----------------------------------------------\n");
  }
  
  // Objects to be returned to R
  arma::sp_mat beta = arma::sp_mat(p, L); // beta
  double *a = Calloc(p, double); //Beta from previous iteration
  NumericVector loss(L);
  IntegerVector iter(L);
  IntegerVector n_reject(L);
  
  double l1, l2, shift;
  double max_update, update, thresh; // for convergence check
  int i, j, jj, l, violations, lstart;
  int *e1 = Calloc(p, int); // ever active set
  double *r = Calloc(n, double);
  
  // hypothesis testing, differencing
  double *r_diff = Calloc(n, double);
  double *z_prev = Calloc(p, double);
  double *var = Calloc(p, double);
  int *start_pos = Calloc(p, int);
  
  for (i = 0; i < n; i++) r[i] = y[i];
  
  int resid_diff_check_number = 0;
  
  double sumResid = sum(r, n);
  loss[0] = gLoss(r,n);
  thresh = eps * loss[0] / n;
  int steps = 0, stepsum = 0; 
  
  // set up lambda
  if (user == 0) {
    if (lam_scale) { // set up lambda, equally spaced on log scale
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
  } else {
    lstart = 0;
    lambda = Rcpp::as<NumericVector>(lambda_);
  }
  
  // Path
  for (l = lstart; l < L; l++) {
    if(verbose) {
      // output time
      char buff[100];
      time_t now = time (0);
      strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
      Rprintf("Lambda %d. Now time: %s\n", l, buff);
    }
    if (l != 0) {
      // Check dfmax
      int nv = 0;
      for (j = 0; j < p; j++) {
        if (a[j] != 0) nv++;
      }
      if (nv > dfmax) {
        for (int ll=l; ll<L; ll++) iter[ll] = NA_INTEGER;
        Free_memo(a, r, e1);
        return List::create(beta, center, scale, lambda, loss, iter, n_reject, Rcpp::wrap(col_idx));
      }
    } 
    
    while(iter[l] < max_iter){
      // Coordinate descent
      while(iter[l] < max_iter) {
        iter[l]++;
        
        //solve lasso over ever-active set
        max_update = 0.0;              
        
        for (j = 0; j < p; j++) {
          if (e1[j]) {
            jj = col_idx[j];
            z[j] = crossprod_resid(xMat, r, sumResid, row_idx, center[jj], scale[jj], n, jj) / n + a[j];
            l1 = lambda[l] * m[jj] * alpha;
            l2 = lambda[l] * m[jj] * (1-alpha);
            beta(j, l) = lasso(z[j], l1, l2, 1);
            
            shift = beta(j, l) - a[j];
            if (shift !=0) {
              // compute objective update for checking convergence
              //update =  z[j] * shift - 0.5 * (1 + l2) * (pow(beta(j, l), 2) - pow(a[j], 2)) - l1 * (fabs(beta(j, l)) -  fabs(a[j]));
              update = pow(beta(j, l) - a[j], 2);
              if (update > max_update) {
                max_update = update;
              }
              
              update_resid(xMat, r, shift, row_idx, center[jj], scale[jj], n, jj); // update r
              resid_diff_check_number++;
              sumResid = sum(r, n); //update sum of residual
              a[j] = beta(j, l); //update a
            }
          }
        }
        // Check for convergence
        if (max_update < thresh) break;
      }
      // Adding new values to r_diff to get difference 
      for  (int j = 0; j < n; j++) r_diff[j] += r[j]; 
      
      // Scan for violations in inactive set
      violations = check_inactive_set(e1, z, xMat, row_idx, col_idx, center, scale, a, lambda[l], sumResid, alpha, r, m, n, p, steps, stepsum,
                                      r_diff, z_prev, var, start_pos); 
      
      // Setting r_diff to reflect current values
      for  (int j = 0; j < n; j++) r_diff[j] = -r[j]; 
      
      if (violations==0) {
        loss[l] = gLoss(r, n);
        break;
      }
    }
  }
  
  Rprintf("\n Avg steps: %f %d\n", ((double)stepsum)/steps/n, n);
  Rprintf("\n resid diff num: %d\n", resid_diff_check_number);
  
  Free_memo(a, r, e1);
  return List::create(beta, center, scale, lambda, loss, iter, n_reject, Rcpp::wrap(col_idx));
}
