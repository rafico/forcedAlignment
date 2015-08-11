#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include <sys/time.h>

#include <omp.h>

#include "jsgd.h"



/* random value between 0 and n - 1 */
static long jrandom_l(jsgd_params_t *params, long n) {
  long res; 
  res = rand_r(&params->rand_state); 
  res ^= (long)rand_r(&params->rand_state) << 31; 
  return res % n;
}

/* generates the k first elts of a permutation of size n, returns
   perm[k]. perm still needs to be of size n to provide buffer space */
static void jrandom_perm(jsgd_params_t *params, long n, long k, int *perm) {
  int i, j; 

  if(k < n / 10) {
    
    for(i = 0; i < k; i++) {
      int ybar;
    draw_ybar:
      ybar = jrandom_l(params, n); 
      for(j = 0; j < i; j++) if(ybar == perm[j]) goto draw_ybar;          
      perm[i] = ybar; 
    }

  } else {
    perm[0] = 0;
    for(i = 0; i < n; i++) {
      long j = jrandom_l(params,i+1);
      perm[i] = perm[j];
      perm[j] = i;
    }
  } 
}


/*******************************************************************
 * Higher-level ops
 *
 */


/* keep W normalization factor up-to-date */
static void renorm_w_factor(float *w, int d, float *fw) {
  if(1e-4 < *fw && *fw < 1e4) return;
  vec_scale(w, d, *fw);                    
  *fw = 1;
}

static void renorm_w_matrix(float *w, int d, int nclass, float *w_factors) {
  int j; 
  for(j = 0; j < nclass; j++) {
    vec_scale(w + d * j, d, w_factors[j]);
    w_factors[j] = 1.0;
  }
}

#define NEWA(type,n) (type*)malloc(sizeof(type)*(n))
#define NEWAC(type,n) (type*)calloc(sizeof(type),(n))
#define NEW(type) NEWA(type,1)

void jsgd_compute_scores(const x_matrix_t *x, int nclass, 
                         const float *w, const float *bias, 
                         float bias_term, 
                         int threaded, 
                         float *scores) {

  if(threaded) 
    x_matrix_matmul_thread(x, w, nclass, scores);    
  else
    x_matrix_matmul(x, w, nclass, scores);

  int i,j;
  for(i = 0; i < x->n; i++) {
    float *scores_i = scores + i * nclass;
    for(j = 0; j < nclass; j++) 
      scores_i[j] += bias[j] * bias_term;
  }

}

static double compute_accuracy_top1(const float *scores, const int *y, int nclass, int n, 
                                    void *ignored) {  
  double accu = 0; 
  int i,j;

  for(i = 0; i < n; i++) {
    const float *scores_i = scores + i * nclass;
    
    /* find rank of correct class */
    int nabove = 0, neq = 0; 
    float class_score = scores_i[y[i]];

    for(j = 0; j < nclass; j++) {      
      float score = scores_i[j];
      if(score > class_score) nabove ++; 
      else if(score == class_score) neq ++; 
    }
    int rank = nabove + neq / 2; /* a synthetic rank */

    if(rank == 0) accu += 1; 

  }

  return accu / n;
}
  

static double compute_accuracy(const x_matrix_t *x, const int *y, int nclass, 
                               const float *w, const float *bias, float bias_term,                               
                               int threaded, 
                               jsgd_accuracy_function_t *accuracy_function, 
                               void * accuracy_function_arg) {
  float *scores = NEWA(float, nclass * x->n); 
  
  jsgd_compute_scores(x, nclass, w, bias, bias_term, threaded, scores);
  
  double score; 

  if(!accuracy_function) 
    accuracy_function = &compute_accuracy_top1; 

  score = (*accuracy_function)(scores, y, nclass, x->n, accuracy_function_arg); 

  free(scores);

  return score;
}




/*******************************************************************
 * reporting
 *
 */


static double getmillisecs() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return tv.tv_sec*1e3 +tv.tv_usec*1e-3;
}


static double evaluation(int nclass,
                       const x_matrix_t *x,
                       const int *y,
                       float *w,
                       float *bias, 
                       jsgd_params_t *params, 
                       long t) {

  double t0 = getmillisecs(); 
  double elapsed_t = (t0 - params->t0) * 1e-3;
  double train_accuracy = -1, valid_accuracy = -1; 
  
  if(params->verbose)
    printf("Evaluation at epoch %ld (%ld samples, %.3f s), %ld vector ops\n",
           t / x->n, t, elapsed_t, params->ndp + params->nmodif);
  
  if(params->compute_train_accuracy) {
    train_accuracy = compute_accuracy(x, y, nclass, w, bias, params->bias_term, params->n_thread, 
                                      params->accuracy_function, params->accuracy_function_arg); 
    if(params->verbose) printf("train_accuracy = %.5f\n", train_accuracy); 
  }
  
  if(params->valid) {
    valid_accuracy = compute_accuracy(params->valid, params->valid_labels, nclass, w, bias, params->bias_term, 
                                      params->n_thread, 
                                      params->accuracy_function, params->accuracy_function_arg); 
    
    if(params->verbose) printf("valid_accuracy = %.5f\n", valid_accuracy); 

  }      

  params->t_eval += (getmillisecs() - t0) * 1e-3; 

  long se = t / (x->n * params->eval_freq); 
  if(se < params->na_stat_tables) {
    if(params->valid_accuracies) params->valid_accuracies[se] = valid_accuracy;              
    if(params->train_accuracies) params->train_accuracies[se] = train_accuracy;              
    if(params->times) params->times[se] = elapsed_t;      
    if(params->ndotprods) params->ndotprods[se] = params->ndp; 
    if(params->nmodifs) params->nmodifs[se] = params->nmodif;       
  }  

  return valid_accuracy; 
}




void jsgd_params_set_default(jsgd_params_t *params) {
  memset(params, 0, sizeof(jsgd_params_t)); /* most defaults are ok at 0 */

  params->algo = JSGD_ALGO_OVR; 
  params->stop_valid_threshold = -1; 
  params->n_epoch = 10;
  
  params->lambda = 1e-4; 
  params->bias_term = 0.1;
  params->eta0 = 0.1;
  params->beta = 0;
  params->temperature = 1;
  params->bias_learning_rate = 0.01;
  params->eta_nepoch= 5;

  params->avg = JSGD_AVG_NONE;
  params->rho = 1;

  params->eval_freq = 10; 
  
  params->t_block = 16; 
  params->n_wstep = 32; 
  params->d_step = 1024;

  params->biasopt_netas = 9;           
  params->biasopt_etafactor =10;  
  params->biasopt_nsample_eta = 10000;     
  params->biasopt_n_epoch = 200;        
		
}


/*******************************************************************
 * Learning, simple version
 *
 */


static void learn_epoch_ovr(int nclass,
                            const x_matrix_t *x,
                            const int *y,
                            float *w, 
                            float *bias, 
                            jsgd_params_t *params, 
                            long t0,
			    const int* perm); 

static void compute_self_dotprods(const x_matrix_t *x, jsgd_params_t *params); 

static int learn_slice_logloss(int nclass,
                               const x_matrix_t *x,
                               const int *y,
                               float *w, 
                               float *bias, 
                               jsgd_params_t *params, 
                               long t0, long t1, const int* perm); 

static int learn_slice_stiff(int nclass,
			      const x_matrix_t *x,
			      const int *y,
			      float *w, 
			      float *bias, 
			      jsgd_params_t *params, 
			     long t0, long t1, const int* perm); 


static int learn_slice_mul2(int nclass, 
			     const x_matrix_t *x,
			     const int *y,
			     float *w, 
			     float *bias, 
			     jsgd_params_t *params, 
			    long t0, long t1, const int* perm); 

static int learn_slice_sqr(int nclass, 
			   const x_matrix_t *x,
			   const int *y,
			   float *w, 
			   float *bias, 
			   jsgd_params_t *params, 
			   long t0, long t1, const int* perm); 

static int learn_slice_others(int nclass,
                               const x_matrix_t *x,
                               const int *y,
                               float *w, 
                               float *bias, 
                               jsgd_params_t *params, 
			      long t0, long t1, const int* perm); 


static void find_eta0(int nclass,
                      const x_matrix_t *x,
                      const int *y,
                      float *w, 
                      float *bias, 
                      jsgd_params_t *params);

static void check_gradient(const x_matrix_t *x,
			   const int *y,
			   const float *w,  
			   const float* bias,
			   int nclass, 
			   int i,
			   const float* gradient, 
			   float* bgrad, 
			   jsgd_params_t *params);

static float compute_cost(const x_matrix_t *x, const int *y, int nclass, 
                         const float *w, const float *bias, 
			  const jsgd_params_t *params);

static void fine_tune_bias(const x_matrix_t *x, const int *y, int nclass, 
			   const float *w, float *bias, 
			   const jsgd_params_t *params);

int jsgd_train(int nclass,
               const x_matrix_t *x,
               const int *y,
               float *w, 
               float *bias, 
               jsgd_params_t *params) {

  const long n = x->n, d = x->d; 

  params->rand_state = params->random_seed;

  float *best_w = NULL, *best_bias = NULL;

  if(params->avg == JSGD_AVG_NONE){
    if(params->valid) { /* we don't necessarily keep the last result */
      /* make temp buffers for current result */    
      best_w = w;  w = NEWA(float, nclass * d); 
      best_bias = bias;  bias = NEWA(float, nclass);
    }
    params->wavg = w;
    params->bavg = bias;
  }  
  else{
    params->wavg = NEWA(float,nclass*d);
    params->bavg = NEWA(float,nclass);
    memcpy(params->wavg,w,sizeof(float) * nclass * d);
    memcpy(params->bavg,bias,sizeof(float) * nclass);

    if(params->valid) { /* we don't necessarily keep the last result */
      /* make temp buffers for current result */    
      best_w = w;  w = NEWA(float, nclass * d); 
      best_bias = bias;  bias = NEWA(float, nclass);
    }
  }


  if(params->verbose) {
    printf("jsgd_learn: %ld training (%s) examples in %ldD from %d classes\n", 
           n, 
           params->algo ==  JSGD_ALGO_OVR  ? "OVR" : 
           params->algo ==  JSGD_ALGO_MUL  ? "MUL" : 
           params->algo ==  JSGD_ALGO_MUL2  ? "MUL2" : 
           params->algo ==  JSGD_ALGO_RNK  ? "RNK" : 
           params->algo ==  JSGD_ALGO_WAR  ? "WAR" : 
	   params->algo ==  JSGD_ALGO_LOG  ? "LOG" : 
	   params->algo ==  JSGD_ALGO_STF ? "STF":
           "!! unknown",
           d, nclass); 
    if(params->valid) 
      printf("  validation on %ld examples every %d epochs (criterion: %s)\n", 
             params->valid->n, params->eval_freq, 
             params->accuracy_function ? "custom" : "top-1 accuracy");

    if(params->valid && params->stop_valid_threshold > -1) 
      printf("stopping criterion: after %ld epochs or validation accuracy below best + %g\n",
             params->n_epoch, params->stop_valid_threshold);
    else
      printf("stopping after %ld epochs\n", params->n_epoch);
      
  } 

  params->ndp = params->nmodif = 0; 
  params->t0 = getmillisecs();
  params->best_valid_accuracy = 0;

  if(!params->use_input_w) {
    memset(w, 0, sizeof(w[0]) * nclass * d);
    memset(bias, 0, sizeof(bias[0]) * nclass);
  }

  if(params->algo == JSGD_ALGO_OVR && params->beta == 0) {
    params->beta = (int)sqrt(nclass); 
    if(params->verbose) printf("no beta provided, setting beta = %d\n", params->beta); 
  }

  if(params->eta0 == 0) {
    if(params->verbose) printf("no eta0 provided, searching...\n"); 
    find_eta0(nclass, x, y, w, bias, params); 
    if(params->verbose) printf("keeping eta0 = %g\n", params->eta0); 
  }
  
  if(params->algo  == JSGD_ALGO_OVR && params->use_self_dotprods) {
    compute_self_dotprods(x, params);     
  }

  long t; 
  int eval_valid = 0; /* is the current evaluation up-to-date? */

  int* perm = NEWA(int,n);
 


  for(t = 0; t < params->n_epoch * n; t += n) {

    if((t / n) % params->eval_freq == 0) {
      double valid_accuracy = evaluation(nclass, x, y, params->wavg, params->bavg, params, t); 
      eval_valid = 1;

      if(valid_accuracy < params->best_valid_accuracy + params->stop_valid_threshold) {
        if(params->verbose)
          printf("not improving enough on validation: stop\n"); 
        break;
      }           

      if(valid_accuracy > params->best_valid_accuracy) {
        memcpy(best_w, params->wavg, sizeof(float) * nclass * d);
        memcpy(best_bias, params->bavg, sizeof(float) * nclass);        
        params->best_epoch = t / n; 
        params->best_valid_accuracy = valid_accuracy;
      }      

      
      if(params->eval_objective) {
	double obj = compute_cost(x,y,nclass, params->wavg, params->bavg, params);
	if(params->verbose) printf("Objective: %f\n",obj);
	params->objectives[t / n] = obj;
      }
    }

    if(params->verbose > 1) {
      printf("Epoch %ld (%.3f s), %ld vector operations\r",
             t / x->n, (getmillisecs() - params->t0) * 1e-3, 
             params->ndp + params->nmodif);
      fflush(stdout);
    }

    //Random permutation.
    jrandom_perm(params,n, n, perm);

    switch(params->algo) {     
    case JSGD_ALGO_OVR: 
      learn_epoch_ovr(nclass, x, y, w, bias, params, t, perm);
      break;
    case JSGD_ALGO_LOG:
      learn_slice_logloss(nclass,x,y,w,bias,params,t,t+n , perm);
      break;
    case JSGD_ALGO_STF:
      learn_slice_stiff(nclass,x,y,w,bias,params,t,t+n , perm);
      break;
    case JSGD_ALGO_MUL2:
      learn_slice_mul2(nclass,x,y,w,bias,params,t,t+n, perm);
      break;
    case JSGD_ALGO_SQR:
      learn_slice_sqr(nclass,x,y,w,bias,params,t,t+n,perm);
    case JSGD_ALGO_MUL: case JSGD_ALGO_RNK: case JSGD_ALGO_WAR: 
      learn_slice_others(nclass, x, y, w, bias, params, t, t + n, perm);
      break;
      
    default:       
      assert(!"not implemented");    
    }
    eval_valid = 0;
  }

  if(!eval_valid) {  /* one last evaluation */
    double valid_accuracy = evaluation(nclass, x, y, params->wavg, bias, params, t);     
    if(valid_accuracy > params->best_valid_accuracy) {
      memcpy(best_w, params->wavg, sizeof(w[0]) * nclass * d);
      memcpy(best_bias, params->bavg, sizeof(bias[0]) * nclass);        
      params->best_epoch = t / n; 
      params->best_valid_accuracy = valid_accuracy;
    }  
  }

  if(params->fine_tune_bias){
    fine_tune_bias(x,y,nclass,w,bias,params);
  }

  

  if(params->verbose && best_w) 
    printf("returning W obtained at epoch %d (valid_accuracy = %g)\n", 
           params->best_epoch, params->best_valid_accuracy); 
  params->niter = t; 

  free(params->self_dotprods);
  params->self_dotprods = NULL; 
  if(best_w) {free(w); free(bias); }  

  if(params->avg != JSGD_AVG_NONE){
    if(!best_w){
      memcpy(w,params->wavg, sizeof(float) * nclass * d); 
      memcpy(bias,params->bavg, sizeof(float) * nclass);
    }
    //otherwise, best_w already contains the average
  }
  else{
    params->wavg = NULL; //To avoid memory errors in python
  }

  return t;
}

  static void update_average(const float* w, const float* b,int w_size, int nclass, int t, int tmax, float wfactor, float* wfactors,jsgd_params_t *params);

/*******************************************************************
 * One Versus Rest implementation
 *
 * There are actually 3 implementations, corresponding to different
 * levels of optimization. 
 */


/* 1 OVR step: update all w's from 1 vector */
static void learn_ovr_step(int nclass,
                           const x_matrix_t *x,
                           const int *y,
                           float *w, 
                           float *bias, 
                           jsgd_params_t *params, 
                           long t, float *w_factors, 
                           const int *ybars, int nybar,const int* perm);

/* more complicated implementation */

static void learn_epoch_ovr_blocks(int nclass,
                                   const x_matrix_t *x,
                                   const int *y,
                                   float *w, 
                                   float *bias, 
                                   jsgd_params_t *params, 
                                   long t0, 
                                   const int *ybars, int ldybar,const int* perm); 


#if 1
static void learn_epoch_ovr(int nclass,
                            const x_matrix_t *x,
                            const int *y,
                            float *w, 
                            float *bias, 
                            jsgd_params_t *params, 
                            long t0, const int* perm) {

  assert(params->beta <= nclass - 1);    /* else  infinite loop */
  
  const long n = x->n, d = x->d; 
  int i, k;

  /* We choose in advance which classes to sample. This makes it
   * possible to reproduce results across implementations.  (NB. we
   * sample classes instead of examples, because the outer loop is on
   * the examples, to factor the cache accesses and/or decompression
   * steps). */

  int ldybar = params->beta + 1; 
  int *ybars = NEWA(int, ldybar * n + n);
  
  long t, t1 = t0 + n; 

  for(t = t0; t < t1; t++) {   
    long i = perm[t % n]; 
    //printf("%d\n",i);
    int yi = y[i];

    int *seen = ybars + ldybar * (t - t0); 

    /* draw beta ybars */
    seen[0] = yi; 

    /* generates values between 0 and nclass - 2 */
    jrandom_perm(params, nclass - 1, params->beta, seen + 1); 
  
   
    /* but yi is already taken, so shift up */
    for(k = 1; k <= params->beta; k++) 
      if(seen[k] >= yi) seen[k]++; 

  }

  double tt0 = getmillisecs(); 

  if(params->avg!= JSGD_AVG_NONE || params->n_thread <=1) {
    /* simple implementation: handle 1 example at a time */

    /* W is encoded as W * diag(w_factors). This avoids some vector
       multiplications */
    float *w_factors = NEWA(float, nclass); 
      
    for(i = 0; i < nclass; i++) w_factors[i] = 1; 
   
    for(t = t0; t < t1; t++) {
      int *ybars_i = ybars + ldybar * (t - t0);       
      learn_ovr_step(nclass, x, y, w, bias, params, t, w_factors, ybars_i, ldybar,perm);       
    }
    renorm_w_matrix(w, d, nclass, w_factors);
    free(w_factors); 

  } else {
    /* more complex implementation: the examples are handled by batches (blocks) */    
    learn_epoch_ovr_blocks(nclass, x, y, w, bias, params, t0, ybars, ldybar,perm); 
  }

  if(params->verbose > 2) printf("OVR epoch t = %.3f ms\n", getmillisecs() - tt0);

  free(ybars);
}

#else
static void learn_epoch_ovr(int nclass,
                            const x_matrix_t *x,
                            const int *y,
                            float *w, 
                            float *bias, 
                            jsgd_params_t *params, 
                            long t0, const int* perm) {
  int n = x->n,d=x->d;
  //printf("Parameters: lambda: %g | eta0: %g | bias term: %g\n",params->lambda,params->eta0,params->bias_term);

  //compute objective (to remove)
  float* scores = NEWA(float,n*nclass);
  x_matrix_matmul(x,w,nclass,scores);
  int i,j;
  double m=0;
  for(i=0; i<n; ++i){
    for(j=0;j<nclass; ++j){
      int sense = y[i]==j ? 1 : -1;
      double a = 1 - sense * (scores[i*nclass] + params->bias_term * bias[j]);
      a = a>=0? a : 0;
      m+=a;
    }
  }
  free(scores);
  m = m/n + params->lambda/2 * vec_sqnorm(w,d*nclass);
  printf("OBJECTIVE=%g\n",m);



  int t,k;

  //float *w_factors = NEWA(float, nclass);  
  float wfactor = 1;

  //for(k = 0; k < nclass; k++) w_factors[k] = 1; 

  for(t=t0;t<t0+n; ++t){
    int i = perm[t%n];

    double eta = params->fixed_eta ? params->eta0 : params->eta0 / (1 + params->lambda * params->eta0 * t);
    //if(params->avg != JSGD_AVG_NONE) eta = pow(eta,0.75);
    double fw =  (1 - eta * params->lambda);

    for(k=0;k<nclass;++k){
      float* wk = w + k*d;
      double score = x_matrix_dotprod(x, i, wk) * wfactor + params->bias_term * bias[k];

      int sense = k==y[i]? +1 : -1;

      double v = 1-sense*score;
      if(v>0){
	x_matrix_addto(x,i,(sense * eta)/wfactor,wk);
	bias[k] += sense * params->bias_learning_rate * eta * params->bias_term;
	params->nmodif++; 
      }
    }
    wfactor *= fw;
  }
  vec_scale(w, d*nclass, wfactor); 
  wfactor=1;
  


}

#endif



/* 1 OVR step: update all w's from 1 vector */
static void learn_ovr_step(int nclass,
                           const x_matrix_t *x,
                           const int *y,
                           float *w, 
                           float *bias, 
                           jsgd_params_t *params, 
                           long t, float *w_factors, 
                           const int *ybars, int nybar,
			   const int* perm) {

  const long n = x->n, d = x->d; 

  float bias_term = params->bias_term;
  
  long i = perm[t % n];
  int yi = y[i];
  
  double eta = params->fixed_eta ? params->eta0 : params->eta0 / (1 + params->lambda * params->eta0 * t);
  if(params->avg != JSGD_AVG_NONE) eta = pow(eta,0.75);

  double fw = 1 - eta * params->lambda;    
  
  if(params->verbose > 2) printf("iteration %ld, sample %ld, label %d, eta = %g\n", t, i, yi, eta);
  
  int k;
  for(k = 0; k < nybar; k++) {
    int ybar = ybars[k];     
    float *w_ybar = w + ybar * d; 
    
    float sense = y[i] == ybar ? 1.0 : -1.0; 
    
    double score = x_matrix_dotprod(x, i, w_ybar) * w_factors[ybar] + bias_term * bias[ybar];
    w_factors[ybar] *= fw;  
    
    if(sense * score < 1) { /* inside margin or on wrong side */
      x_matrix_addto(x, i, eta * sense / w_factors[ybar], w_ybar); 
      bias[ybar] += params->bias_learning_rate * eta * sense * bias_term;
      params->nmodif++; 
    }
    
    renorm_w_factor(w_ybar, d, &w_factors[ybar]);  
  }
        
  update_average(w,bias,d,nclass,t+1,n*params->n_epoch,0,w_factors,params);

  params->ndp += nybar;

}


/***************************************************************************
 * blocked code (hairy!)
 */


/* 1 OVR step (transposed): all w's from a set of vectors */

static void learn_ovr_step_w(int nclass,
                             const x_matrix_t *x,
                             const int *y,
                             float *w_yi, 
                             float *bias_yi_io, 
                             const jsgd_params_t *params, 
                             int yi,
                             long t0, const int * ts, int nt, 
                             int *ndp_io, int *nmodif_io,
			     const int* perm) {
  
  const long n = x->n, d = x->d; 

  float bias_term = params->bias_term;

  if(params->verbose > 2) printf("   handling W, label %d, %d ts\n", yi, nt);

  float bias_yi = *bias_yi_io; 
  float w_factor = 1.0;
  int nmodif = 0; 

  int k;
  for(k = 0; k < nt; k++) {
    long t = t0 + ts[k];
    long i = perm[t % n]; 
    float sense = y[i] == yi ? 1.0 : -1.0;

    double eta = params->fixed_eta ? params->eta0 : params->eta0 / (1 + params->lambda * params->eta0 * t);
    double fw = 1 - eta * params->lambda;    

    double score = x_matrix_dotprod(x, i, w_yi) * w_factor + bias_term * bias_yi; 

    if(params->verbose > 3) printf("      score with x_%ld = %g (%d modifs)\n", i, score, nmodif);
  
    w_factor *= fw;     

    if(sense * score < 1) {
      x_matrix_addto(x, i, sense * eta / w_factor, w_yi); 
      bias_yi += params->bias_learning_rate * sense * eta * bias_term;
      nmodif ++; 
    }    

    renorm_w_factor(w_yi, d, &w_factor); 
  }

  vec_scale(w_yi, d, w_factor); 

  *bias_yi_io = bias_yi; 
  *ndp_io += nt; 
  *nmodif_io += nmodif; 
}

  


/* same, all dot products are precomputed */
static void learn_ovr_step_w_dps(int nclass,
                                 const x_matrix_t *x,
                                 const int *y,
                                 float *w_yi, 
                                 float *bias_yi_io, 
                                 const jsgd_params_t *params, 
                                 int yi,
                                 long t0, const int * ts, int nt, 
                                 const float *w_dps, const float *self_dps, int ldsd,
                                 int *correction_is, float *correction_terms, 
                                 float *w_factor_out, 
                                 int *ndp_io, int *nmodif_io,
				 const int* perm) {
  
  const long n = x->n; 

  float bias_term = params->bias_term;

  if(params->verbose > 2) printf("  DPS handling W, label %d, %d ts\n", yi, nt);

  float bias_yi = *bias_yi_io; 
  float w_factor = 1.0;
  int nmodif = 0; 
  
  int i0 = t0 % n;
  int k;
  for(k = 0; k < nt; k++) {
    long t = t0 + ts[k];
    long i = perm[t % n]; 
    float sense = y[i] == yi ? 1.0 : -1.0;

    double eta = params->fixed_eta ? params->eta0 : params->eta0 / (1 + params->lambda * params->eta0 * t);
    double fw = 1 - eta * params->lambda;    

    /* reconstruct score */
    double score = w_dps[k];
    int c;
    for(c = 0; c < nmodif; c++) 
      score += self_dps[(correction_is[c] - i0) + ldsd * (i - i0)] * correction_terms[c];

    score = score * w_factor + bias_term * bias_yi; 

    if(params->verbose > 3) printf("      score with x_%ld = %g (%d modifs)\n", i, score, nmodif);
  
    w_factor *= fw;     

    if(sense * score < 1) {
      correction_is[nmodif] = i; 
      correction_terms[nmodif] = sense * eta / w_factor;
      bias_yi += params->bias_learning_rate * sense * eta * bias_term;
      nmodif ++; 
    }    

  }

  *w_factor_out = w_factor; 
  *bias_yi_io = bias_yi; 
  *ndp_io += nt; 
  *nmodif_io += nmodif; 
}


/* 
 * input: 
 * 
 * vals(n, m)   st. 0 <= v(i, j) < nval
 *
 * output: 
 *
 * js(m * n) 
 * begins(nval + 1) 
 *
 * j in js(begins(w): begins(w + 1)) iff w in vals(:, j)
 * 
 */

static void make_histogram_with_refs(int n, int m, 
                                     const int *vals, 
                                     int nval,                           
                                     int *js,
                                     int *begins) {
  memset(begins, 0, (nval + 1) * sizeof(int)); 
  begins++; 
  int i, j;

  /* first make histogram */
  for(i = 0; i < m * n; i++) 
    begins[vals[i]] ++; 

  /* cumulative sum */
  int accu = 0;
  for(i = 0; i < nval; i++) {
    int b = begins[i];
    begins[i] = accu; 
    accu += b; 
  }
  assert(accu == m * n); 
  /* now begins[i] contains offset in js where to write values */  

  const int *vp = vals;
  for(j = 0; j < m ; j++) 
    for(i = 0; i < n; i++) 
      js[begins[*vp++]++] = j; 

  /* now all values are written so begins[v] contains end of segment
     v, but we did begins++ so the orginial array indeed contains the
     beginning */
  assert(begins[nval - 1] == n * m);   
}



static void learn_epoch_ovr_blocks(int nclass,
                                   const x_matrix_t *x,
                                   const int *y,
                                   float *w, 
                                   float *bias, 
                                   jsgd_params_t *params, 
                                   long t0, 
                                   const int *ybars, int ldybar, 
				   const int* perm) {
  const long n = x->n, d = x->d; 
  long t_block = params->t_block;   
  int ngroup = (n + params->t_block - 1) / params->t_block; 
  long ldg = params->t_block * params->t_block; 

  double tt0 = getmillisecs();  

  /* convert the ybar arrays to arrays that are indexed by W */
  
  int *all_ts = NEWA(int, (ldybar * t_block) * ngroup);
  int *all_ts_begins = NEWAC(int, (nclass + 1) * ngroup);
  
  long ti; 
  for(ti = 0; ti < ngroup; ti++) {
    int i0 = ti * params->t_block; 
    int i1 = (ti + 1) * params->t_block; 
    if(i1 > n) i1 = n; 
    int *ts = all_ts + ti * (ldybar * t_block); 
    int *ts_begins = all_ts_begins + ti * (nclass + 1); 
    
    make_histogram_with_refs(ldybar, i1 - i0, ybars + ldybar * ti * t_block, nclass, 
                             ts, ts_begins); 

    /* for ts_begins[j] <= i < ts_begins[j + 1], x(:, t0 + ts[i]) interacts with W(:, j) */
    
  }     

  int n_wstep = params->n_wstep; 
  if(!n_wstep) 
    n_wstep = omp_get_max_threads();    


  int ndp = 0, nmodif = 0; 
  
  int wi; 
#pragma omp parallel for  reduction(+: ndp, nmodif)
  for(wi = 0; wi < n_wstep; wi++) {
    int wa = wi * nclass / n_wstep, wb = (wi + 1) * nclass / n_wstep;
    
    int ti; 
    for(ti = 0; ti < ngroup; ti++) {
      int i0 = ti * params->t_block; 
      int i1 = (ti + 1) * params->t_block; 
      if(i1 > n) i1 = n; 

      int *ts = all_ts + ti * (ldybar * t_block); 
      int *ts_begins = all_ts_begins + ti * (nclass + 1); 
      
      int nnz = ts_begins[wb] - ts_begins[wa]; 
      
      if(params->verbose > 2) 
        printf("handling block W %d:%d * T %d:%d, %d interactions (%.2f %%)\n", 
               wa, wb, i0, i1, nnz, nnz * 100.0 / ((wb - wa) * (i1 - i0))); 
      
      if(!params->use_self_dotprods) {
        /* second implementation */
        int yi; 
        for(yi = wa; yi < wb; yi++) 
          learn_ovr_step_w(nclass, x, y, w + yi * d, &bias[yi], 
                           params, yi, t0 + i0, 
                           ts + ts_begins[yi], 
                           ts_begins[yi + 1] - ts_begins[yi], 
                           &ndp, &nmodif,perm);      
        
      } else {
          
        /* third implementation */

        /* Precompute dot products of x_i's with W_j's */

        /* register which dot products must be computed */
        x_matrix_sparse_t w_dps; 
        
        x_matrix_sparse_init(&w_dps, i1 - i0, wb - wa, nnz); 
        
        int k = 0, yi, i;
        for(yi = wa; yi < wb; yi++) {        
          w_dps.jc[yi - wa] = k;
          for(i = ts_begins[yi]; i < ts_begins[yi + 1]; i++) 
            w_dps.ir[k++] = ts[i] + i0;               
        }
        w_dps.jc[wb - wa] = k;
        
        /* compute actual dot products */
        x_matrix_matmul_subset(x, &w_dps, w + wa * d, wb - wa, params->d_step); 
        
        /* dot products of x_i and x_j's */
        float *self_dps = params->self_dotprods + ti * ldg;
        
        /* matrix which registers the updates to perform */
        x_matrix_sparse_t correction; 
        x_matrix_sparse_init(&correction, i1 - i0, wb - wa, nnz);           

        float *w_factors = NEWA(float, wb - wa); 
        
        /* compute OVR updates */
        int nmodif_i = 0;
        for(yi = wa; yi < wb; yi++) {
          correction.jc[yi - wa] = nmodif_i;             
          learn_ovr_step_w_dps(nclass, x, y, w + yi * d, &bias[yi], 
                               params, yi, t0 + i0, 
                               ts + ts_begins[yi], 
                               ts_begins[yi + 1] - ts_begins[yi], 
                               w_dps.pr + w_dps.jc[yi - wa],                                  
                               self_dps, i1 - i0, 
                               correction.ir + nmodif_i, 
                               correction.pr + nmodif_i, 
                               &w_factors[yi - wa], 
                               &ndp, &nmodif_i,perm);      
        }
        correction.jc[wb - wa] = nmodif_i; 
        nmodif += nmodif_i; 

        if(params->verbose > 2) 
          printf("applying %d W corrections\n", nmodif_i); 
        
        /* apply correction to Ws */          
        x_matrix_addto_sparse(x, &correction, w_factors, w + wa * d, wb - wa, params->d_step);
        
        free(w_factors); 
        
        x_matrix_sparse_clear(&w_dps); 
        x_matrix_sparse_clear(&correction); 
        
      }
      
    }
  }
  free(all_ts); 
  free(all_ts_begins); 

  params->ndp += ndp; 
  params->nmodif += nmodif; 

  if(params->verbose > 2) printf("blocked t = %.3f ms\n", getmillisecs() - tt0);

}



static void compute_self_dotprods(const x_matrix_t *x, jsgd_params_t *params) {
  int n = x->n; 
  int ngroup = (n + params->t_block - 1) / params->t_block; 
  long ldg = params->t_block * params->t_block; 

  if(params->verbose) 
    printf("Computing dot products within training data (%d groups of max size %d) using %.2f MB...\n", 
           ngroup, params->t_block, ngroup * ldg * 4 / (1024.0*1024));        
  
  params->self_dotprods = NEWA(float, ngroup * ldg); 
  
  double t0 = getmillisecs(); 

  int i; 
#pragma omp parallel for 
  for(i = 0; i < ngroup; i++) {
    int i0 = i * params->t_block; 
    int i1 = (i + 1) * params->t_block; 
    if(i1 > n) i1 = n; 
    x_matrix_matmul_self(x, i0, i1, params->self_dotprods + i * ldg); 
  }
  
  if(params->verbose) 
    printf("done in %.3f s\n", (getmillisecs() - t0) * 1e-3); 

}


/*******************************************************************
 * Other algorithms (not OVR) implementation
 *
 */
static void check_loss(const x_matrix_t *x, const int *y,const float *w,  const float* bias, int nclass,int i, float loss, jsgd_params_t *params);

static int learn_slice_logloss(int nclass,
                               const x_matrix_t *x,
                               const int *y,
                               float *w, 
                               float *bias, 
                               jsgd_params_t *params, 
                               long t0, long t1, const int* perm){

  const long n = x->n, d = x->d; 
  float bias_term = params->bias_term;
  float *scores = NULL; 
  float *bgrad = NULL;

  scores = NEWA(float, nclass);
  bgrad = NEWA(float,nclass);
  
  long t;
  for(t = t0; t < t1; t++) {
    
    long i = perm[t % n],j; 
    int yi = y[i];
    
    double eta =  params->fixed_eta ? params->eta0 : params->eta0 / (1 + params->lambda * params->eta0 * t);
    

    x_matrix_matmul_slice(x, i, i + 1, w, nclass, scores); 
    params->ndp += nclass; 
    
    
    float best_score = -1e20;    
    for(j = 0; j < nclass; j++) {
      scores[j] = scores[j] + bias[j] * bias_term;
      if(scores[j] > best_score) {
	best_score = scores[j];       
      }
    }
    float wxyi = scores[yi];

    float sum = 0;
    for(j = 0; j < nclass; ++j) {
      scores[j] = exp(scores[j] - best_score);
      sum += scores[j];
    }

    if(params->check_level==1 || params->check_level==3){
      float loss = log(sum) + best_score - wxyi;
      check_loss(x,y,w,bias,nclass,i,loss,params);
    }

    for(j=0; j<nclass; ++j){
      scores[j] /= sum;
      bgrad[j] = bias_term * scores[j];
    }
    scores[yi] -=1;
    bgrad[yi] -= bias_term;

    //checks
   
    if(params->check_level>=2){
      float* gradient = NEWA(float,d*nclass);
      memset(gradient,0,sizeof(float)*d*nclass);
      for(j = 0; j<nclass; ++j){
	x_matrix_addto(x,i, scores[j],gradient+j*d);
      }
      vec_addto(gradient,params->lambda,w,nclass*d);
      check_gradient(x,y,w,bias,nclass,i,gradient,  bgrad, params);
    }

    //Update regularization
    vec_scale(w,1-eta*params->lambda,d*nclass);

    //Update gradient
    for(j=0; j<nclass; ++j){
      x_matrix_addto(x,i, - eta * scores[j],w + j*d);
      bias[j] -= params->bias_learning_rate * eta * bgrad[j];      
    }
    params->nmodif += 2*nclass;    
    update_average(w,bias,d,nclass,t+1,n*params->n_epoch,1,NULL,params);

  }

  free(scores);
  free(bgrad);

  return 0;
}
static int learn_slice_stiff(int nclass,
			      const x_matrix_t *x,
			      const int *y,
			      float *w, 
			      float *bias, 
			      jsgd_params_t *params, 
			     long t0, long t1,
			     const int* perm){

  const long n = x->n, d = x->d; 
  float bias_term = params->bias_term;
  float *scores = NULL; 
  float *bgrad = NULL;

  scores = NEWA(float, nclass);
  bgrad = NEWA(float,nclass);
  
  long t;
  for(t = t0; t < t1; t++) {
    
    long i = perm[t % n],j; 
    int yi = y[i];
    
    double eta =  params->fixed_eta ? params->eta0 : params->eta0 / (1 + params->lambda * params->eta0 * t);
    

    x_matrix_matmul_slice(x, i, i + 1, w, nclass, scores); 
    params->ndp += nclass; 
    
    
    float best_score = -1e20;    
    for(j = 0; j < nclass; j++) {
      float delta = j==yi ? 0 : 1; 
      scores[j] = scores[j] + bias[j] * bias_term;
      if(scores[j]+delta > best_score) {
	best_score = scores[j]+delta;       
      }
    }
    float wxyi = scores[yi];

    float sum = 0;
    for(j = 0; j < nclass; ++j) {
      float delta = j==yi ? 0 : 1; 
      scores[j] = exp(params->temperature*(scores[j] - best_score + delta));
      sum += scores[j];
    }

    if(params->check_level==1 || params->check_level==3){
      float loss = log(sum)/(x->n*params->temperature) + best_score - wxyi;
      check_loss(x,y,w,bias,nclass,i,loss,params);
    }

    for(j=0; j<nclass; ++j){
      scores[j] /= sum;
      bgrad[j] = bias_term * scores[j];

    }
    scores[yi]-=1;
    bgrad[yi] -= bias_term;

    //checks
   
    if(params->check_level>=2){
      float* gradient = NEWA(float,d*nclass);
      memset(gradient,0,sizeof(float)*d*nclass);
      for(j = 0; j<nclass; ++j){
	x_matrix_addto(x,i, scores[j],gradient+j*d);
      }
      vec_addto(gradient,params->lambda,w,nclass*d);
      check_gradient(x,y,w,bias,nclass,i,gradient,  bgrad, params);
    }



    //Update regularization
    vec_scale(w,1-eta*params->lambda,d*nclass);

    //Update gradient
    for(j=0; j<nclass; ++j){
      x_matrix_addto(x,i, - eta * scores[j],w + j*d);
      bias[j] -= params->bias_learning_rate * eta * bgrad[j];      
    }
    params->nmodif += 2*nclass;    
    update_average(w,bias,d,nclass,t+1,n*params->n_epoch,1,NULL,params);
  }

  free(scores);
  free(bgrad);

  return 0;
}


static int learn_slice_mul2(int nclass,
			     const x_matrix_t *x,
			     const int *y,
			     float *w, 
			     float *bias, 
			     jsgd_params_t *params, 
			    long t0, long t1, const int* perm) {

  const long n = x->n, d = x->d; 
  const float bias_term = params->bias_term;
  float *scores = NEWA(float, nclass);
  
  float w_factor = 1.0;

  long t;
  for(t = t0; t < t1; t++) {

    long i = perm[t % n];
    int yi = y[i];
    float *w_yi = w + d * yi; 
    
    double eta =  params->fixed_eta ? params->eta0 : params->eta0 / (1 + params->lambda * params->eta0 * t);
    double fw = 1 - eta * params->lambda;    

    if(params->verbose > 2) printf("iteration %ld, sample %ld, label %d, eta = %g\n", t, i, yi, eta);

  
    int ybar = -1; 
    /* find worst violation */
    x_matrix_matmul_slice(x, i, i + 1, w, nclass, scores); 
    params->ndp += nclass; 
  
    int j; 
    float best_score = -1e20;    
    for(j = 0; j < nclass; j++) {
      float score = scores[j] * w_factor + bias[j] * bias_term;
      if(fabs(score)>1e15) //problem with eta0-> leads to too high values
	return 1;
      if(j != yi) score += 1;
      if(score > best_score) {
	best_score = score; 
	ybar = j;       
      }
    }
    if(ybar==-1){
      return 1;
    }
    w_factor *= fw;
    if(ybar != yi) {
      float factor = 2*(scores[ybar]-scores[yi] + bias_term*(bias[ybar]-bias[yi]) + 1);
      //float factor = 1;
      x_matrix_addto(x, i, eta*factor / w_factor, w_yi); 
      bias[yi] += eta * factor * bias_term;
      
      float *w_ybar = w + d * ybar; 
      x_matrix_addto(x, i, -eta*factor / w_factor, w_ybar); 
      bias[ybar] -= params->bias_learning_rate * eta * factor * bias_term;
      
      params->nmodif += 2;              
    }
    if(w_factor < 1e-4) {
      vec_scale(w, d * nclass, w_factor); 
      w_factor = 1.0;
    }

    update_average(w,bias,d,nclass, t, n*params->n_epoch,w_factor,NULL,params);
  }
  
  vec_scale(w, d * nclass, w_factor); 
  
  
  free(scores); 

  return 0;
}


static int learn_slice_others(int nclass,
			      const x_matrix_t *x,
			      const int *y,
			      float *w, 
			      float *bias, 
			      jsgd_params_t *params, 
			      long t0, long t1, const int* permutation) {

  
  const long n = x->n, d = x->d; 
  float bias_term = params->bias_term;
  float *scores = NULL; 


  int *perm = NULL; 
  float *lk = NULL; 
 

  if(params->algo == JSGD_ALGO_MUL) 
    scores = NEWA(float, nclass);

  if(params->algo == JSGD_ALGO_WAR) {
    perm = NEWA(int, nclass); 
    lk = NEWA(float, nclass); 
    float accu = 0; 
    int k; 
    for(k = 0; k < nclass; k++) {
      accu += 1.0 / (1 + k); 
      lk[k] = accu;
    }
  }

  float w_factor = 1.0;

  long t;
  for(t = t0; t < t1; t++) {

    long i = permutation[t % n]; 
    int yi = y[i];
    float *w_yi = w + d * yi; 
    
    double eta =  params->fixed_eta ? params->eta0 : params->eta0 / (1 + params->lambda * params->eta0 * t);
    double fw = 1 - eta * params->lambda;    

    if(params->verbose > 2) printf("iteration %ld, sample %ld, label %d, eta = %g\n", t, i, yi, eta);

    int ybar = -1; 

   if(params->algo == JSGD_ALGO_MUL) {

      /* find worst violation */
      x_matrix_matmul_slice(x, i, i + 1, w, nclass, scores); 
      params->ndp += nclass; 
      
      int j; 
      float best_score = -1e20;    
      for(j = 0; j < nclass; j++) {
        float score = scores[j] * w_factor + bias[j] * bias_term;
        if(j != yi) score += 1;
        
        if(score > best_score) {
          best_score = score; 
          ybar = j;       
        }
      }
    } else if(params->algo == JSGD_ALGO_RNK) {
      double score_yi = x_matrix_dotprod(x, i, w_yi) * w_factor + bias_term * bias[yi]; 

      do {
        ybar = jrandom_l(params, nclass); 
      } while(ybar == yi);
      
      float *w_ybar = w + d * ybar; 
      double score_ybar = x_matrix_dotprod(x, i, w_ybar) * w_factor + bias_term * bias[ybar]; 

      double L_tri = 1 - score_yi + score_ybar;
      
      if(!(L_tri > 0)) ybar = yi; /* do nothing */

      params->ndp += 2; 
    } else if(params->algo == JSGD_ALGO_WAR) {
      double score_yi = x_matrix_dotprod(x, i, w_yi) * w_factor + bias_term * bias[yi]; 
      
      int j, k; 
      for(j = 0; j < nclass - 1; j++) perm[j] = j; 
      perm[yi] = nclass - 1;

      for(k = 0; k < nclass - 1; k++) {
        int k2 = jrandom_l(params, nclass - 1 - k) + k; 
        ybar = perm[k2]; 
        perm[k2] = perm[k]; 
        /* perm[k] = ybar; */
        
        float *w_ybar = w + d * ybar; 
        double score_ybar = x_matrix_dotprod(x, i, w_ybar) * w_factor + bias_term * bias[ybar]; 
        
        double L_tri = 1 - score_yi + score_ybar;
        
        if(L_tri > 0) break;
      }
      params->ndp += k; 
      
      if(k == nclass - 1) {
        ybar = yi; /* did not find a violation */
      } else {
        /* recompute eta */
        eta *= lk[(nclass - 1) / (k + 1)]; 
      }
    }

    w_factor *= fw;

   if(ybar != yi) {
     assert(ybar!=-1);
     x_matrix_addto(x, i, eta / w_factor, w_yi); 
     bias[yi] += params->bias_learning_rate * eta * bias_term;
     
     float *w_ybar = w + d * ybar; 
     x_matrix_addto(x, i, -eta / w_factor, w_ybar); 
     bias[ybar] -= params->bias_learning_rate * eta * bias_term;
     
     params->nmodif += 2;              
   }
   if(w_factor < 1e-4) {
     vec_scale(w, d * nclass, w_factor); 
     w_factor = 1.0;
   }

   update_average(w,bias,d,nclass, t, n*params->n_epoch,w_factor,NULL,params);

  }
  
  vec_scale(w, d * nclass, w_factor); 
  
  free(perm); 
  free(scores); 
  free(lk);

  return 0;
}


static int learn_slice_sqr(int nclass,
			   const x_matrix_t *x,
			   const int *y,
			   float *w, 
			   float *bias, 
			   jsgd_params_t *params, 
			   long t0, long t1, const int* permutation){
  
  

}



/*******************************************************************
 * Searching eta0 
 *
 */






static void compute_losses_with_scores(const float* scores, const x_matrix_t *x, const int *y, int nclass, 
				       const float *w, const float *bias, 
				       const jsgd_params_t *params, 
				       float *losses) {
  int n = x->n;
  int i, j; 
  
  switch(params->algo) {
  case JSGD_ALGO_OVR: 
    {    /* sum 1v1 hinge losses */
      int *hist = NEWAC(int, nclass);       
      for(i = 0; i < n; i++) hist[y[i]]++;
      
      float rho = 1.0 / (1 + params->beta); 

      memset(losses, 0, sizeof(float) * n); 

      for(j = 0; j < nclass; j++) {
        for(i = 0; i < n; i++) {          
          float yi, weight; 

          if(y[i] == j) { 
            yi = 1; 
            weight = rho / hist[j]; /* eq(3) of CVPR paper */
          } else {
            yi = -1; 
            weight = (1 - rho) / (n - hist[j]); 
          }

	  weight=1.0/n;

          float L_ovr = 1 - yi * scores[i * nclass + j];
          if(L_ovr < 0) L_ovr = 0; 
          
          losses[i] += weight * L_ovr;
        }
      }
      free(hist); 
    }
    break; 
    
  case JSGD_ALGO_MUL: 
    
    for(i = 0; i < n; i++) {
      double vmax = -10000; 
      for(j = 0; j < nclass; j++) {
        float delta = j == y[i] ? 0.0 : 1.0;
        float v = delta + scores[i * nclass + j]; 
        if(v > vmax) vmax = v;         
      }
      losses[i] = vmax - scores[i * nclass + y[i]]; 
    }
    break; 

  case JSGD_ALGO_MUL2:
    for(i = 0; i < n; i++) {
      double vmax = -10000; 
      for(j = 0; j < nclass; j++) {
        float delta = j == y[i] ? 0.0 : 1.0;
        float v = delta + scores[i * nclass + j]; 
        if(v > vmax) vmax = v;         
      }
      float a = vmax - scores[i * nclass + y[i]];
      losses[i] = a*a; 
    }
    break; 

  case JSGD_ALGO_RNK: 
      for(i = 0; i < n; i++) {
        float loss = 0; 
	for(j = 0; j < nclass; j++) {
	  float delta = j == y[i] ? 0.0 : 1.0;
	  float L_tri = delta + scores[i * nclass + j] - scores[i * nclass + y[i]]; 
	  if(L_tri < 0) L_tri = 0; 
	  loss += L_tri; 
	}
	losses[i] = loss; 
      }
      break; 

  case JSGD_ALGO_LOG:
    for(i = 0; i<n; ++i){
      float loss = 0;
      double smax = -1e20;
      for(j = 0; j<nclass; ++j){
	if(smax < scores[i*nclass + j])
	  smax = scores[i*nclass + j];
      }

      for(j = 0; j<nclass; ++j){
	loss += exp(scores[i*nclass+j]-smax);
      }
      losses[i] = log(loss) + smax - scores[i*nclass + y[i]];
    }
    break;

  case JSGD_ALGO_STF:
    for(i = 0; i<n; ++i){
      float loss = 0;
      double smax = -1e20;
      for(j = 0; j<nclass; ++j){
	float delta = j==y[i] ? 0 : 1;
	if(smax < scores[i*nclass + j] + delta)
	  smax = scores[i*nclass + j] + delta;
      }
      for(j = 0; j<nclass; ++j){
	float delta = j==y[i] ? 0 : 1;
	loss += exp(params->temperature*(scores[i*nclass+j] - smax + delta));
      }
      losses[i] = log(loss)/params->temperature + smax - scores[i*nclass + y[i]];
    }
    break;
  default: 
    assert(!"not implemented");     
  }
}

/* per-example loss */
static void compute_losses(const x_matrix_t *x, const int *y, int nclass, 
                            const float *w, const float *bias, 
                            const jsgd_params_t *params, 
                            float *losses) {
  float *scores = NEWA(float, nclass * x->n);   
  jsgd_compute_scores(x, nclass, w, bias, params->bias_term, params->n_thread, scores);


  compute_losses_with_scores(scores, x,y,nclass,w,bias,params,losses);
  free(scores);
}

/* cost according to eq(1) of paper */

static float compute_cost(const x_matrix_t *x, const int *y, int nclass, 
                         const float *w, const float *bias, 
                         const jsgd_params_t *params) {

  {
    int i;
    for(i = 0; i < x->n; i++) assert(y[i] >= 0 && y[i] < nclass); 
  }

  float *losses = NEWA(float, x->n);   
  compute_losses(x, y, nclass, w, bias, params, losses); 

  float loss = 0; 
  int i; 
  for(i = 0; i < x->n; i++) loss += losses[i];

  if(params->algo!=JSGD_ALGO_OVR){
    //loss/=x->n;
  }

  /* regularization term */
  double sqnorm_W = vec_sqnorm(w, nclass * x->d);

  //sqnorm_W += vec_sqnorm(bias, nclass) * params->bias_term * params->bias_term;

  return loss + 0.5 * params->lambda * sqnorm_W; 
}

static void *memdup(void *a, size_t size) {
  return memcpy(malloc(size), a, size); 
}

#define BIG_NUMBER 1e20

static float eval_eta(int nsubset, float eta0, 
                      int nclass,
                      const x_matrix_t *x,
                      const int *y,
                      float *w_in, 
                      float *bias_in, 
                      jsgd_params_t *params) {
  jsgd_params_t subset_params = *params;
  x_matrix_t subset_x = *x; 
  
  subset_x.n = nsubset; 
  subset_params.eta0 = eta0;
  

  float *w = memdup(w_in, sizeof(float) * nclass * x->d); 
  float *bias = memdup(bias_in, sizeof(float) * nclass); 
  
  int error=0;

  int* perm = NEWA(int,nsubset);

  int i;
  for(i=0; i<params->eta_nepoch;++i){
    jrandom_perm(params,nsubset,nsubset,perm);

    switch(params->algo) {
    
    case JSGD_ALGO_OVR: 
      learn_epoch_ovr(nclass, &subset_x, y, w, bias, &subset_params, i*nsubset ,perm);
      break;
    case JSGD_ALGO_MUL2:
      error = learn_slice_mul2(nclass, &subset_x, y, w, bias, &subset_params, i*nsubset, (i+1)*nsubset,perm);
      break;
    case JSGD_ALGO_MUL: case JSGD_ALGO_RNK: case JSGD_ALGO_WAR: 
      error = learn_slice_others(nclass, &subset_x, y, w, bias, &subset_params,i*nsubset , (i+1)*nsubset,perm);
      break;
    case JSGD_ALGO_LOG:
      error = learn_slice_logloss(nclass, &subset_x, y, w, bias, &subset_params,i*nsubset ,(i+1)* nsubset,perm);
      break;
    case JSGD_ALGO_STF:
      error = learn_slice_stiff(nclass, &subset_x, y, w, bias, &subset_params,i*nsubset , (i+1)*nsubset,perm);
      break;
    default:       
      assert(!"not implemented");    
    }
  
    if(error){//When eta0 is too big, the cost can take very large values. In any case, it should not be kept.
      printf("Error: eta0=%g led to too large values\n",eta0);
      return BIG_NUMBER;
    }
  }

  float cost = compute_cost(&subset_x, y, nclass, w, bias, &subset_params); 

  free(w); 
  free(bias);
  
  return cost; 
}

static void find_eta0(int nclass,
                      const x_matrix_t *x,
                      const int *y,
                      float *w, 
                      float *bias, 
                      jsgd_params_t *params) {

  int nsubset = 10000; /* evaluate on this subset */
  if(nsubset > x->n) nsubset = x->n; 

  float factor = 2;   /* multiply or divide eta by this */

  float eta1 = 1;     /* intial estimate */ 
  float cost1 = eval_eta(nsubset, eta1, nclass, x, y, w, bias, params); 
  while(cost1>=BIG_NUMBER/2){//Find a good starting point
    eta1/=factor;
    cost1 = eval_eta(nsubset, eta1, nclass,x,y,w,bias,params);
  }


  float eta2 = eta1 * factor; 
  float cost2 = eval_eta(nsubset, eta2, nclass, x, y, w, bias, params); 

  if(params->verbose > 1) printf("  eta1 = %g cost1 = %g, eta2 = %g cost2 = %g\n", eta1, cost1, eta2, cost2); 

  if(cost2 > cost1) {
    /* switch search direction */
    float tmp = eta1; eta1 = eta2; eta2 = tmp; 
    tmp = cost1; cost1 = cost2; cost2 = tmp; 
    factor = 1 / factor; 
  }

  /* step eta into search direction until cost increases */

  do {  
    eta1 = eta2; 
    eta2 = eta2 * factor;
    cost1 = cost2;     
    cost2 = eval_eta(nsubset, eta2, nclass, x, y, w, bias, params); 
    if(params->verbose > 1) printf("  eta2 = %g cost2 = %g\n", eta2, cost2); 
  } while(cost1 > cost2);
  
  /* keep smallest */
  params->eta0 = eta1 < eta2 ? eta1 : eta2;  

}

//expensive
 static void check_loss(const x_matrix_t *x, const int *y,const float *w,  const float* bias, int nclass,int i, float loss, jsgd_params_t *params){
   float *losses = NEWA(float, x->n);   
   compute_losses(x,y,nclass,w,bias,params,losses);
   if(fabs(loss-losses[i])>0.00001){
     fprintf(stderr,"Error loss %d does not match %g %g\n",i,loss,losses[i]);
     exit(1);
   }
   free(losses);
}

static void check_gradient(const x_matrix_t *x, 
			   const int *y,
			   const float *w,  
			   const float* bias,
			   int nclass, 
			   int index,
			   const float* gradient,
			   float* bgrad, 
			   jsgd_params_t *params){
  int nchecks = 20;
  int i,j;

  float *losses = NEWA(float,x->n);
  compute_losses(x,y,nclass,w,bias,params,losses);
  float cost_ref = losses[index] + (params->lambda/2) * vec_dotprod(w,w,nclass*x->d);

  float* wp = NEWA(float,x->d*nclass);
  float* bp = NEWA(float,nclass);
  for(i=0; i<nchecks; ++i){
    for(j = 0; j<x->d * nclass; ++j){
      wp[j] = w[j] + gradient[j]/pow(2,i);
      if(j<nclass)
	bp[j] = bias[j] + bgrad[j]/pow(2,i);
    }
  
    compute_losses(x,y,nclass,wp,bp,params,losses);
    float cost =  losses[index] + (params->lambda/2) * vec_dotprod(wp,wp,nclass*x->d);
    
    
    if(cost < cost_ref - 0.000001){
      printf("%d: %g < %g \n",i,cost,cost_ref);
      fprintf(stderr,"Error: there is a problem with the loss\n");
      exit(1);
    }
  }
  free(wp);
  free(bp);
  free(losses);

}

/* Modifies the avg parameter according to the current w */
 static void update_average(const float* w, const float* bias,int d, int nclass, int t, int tmax, float wfactor,float* wfactors,jsgd_params_t *params){
  int w_size = d*nclass;
  if(params->avg == JSGD_AVG_WGT){
    vec_scale(params->bavg,nclass, (t-1.0)/(t+1.0));
    vec_addto(params->bavg,2.0/(t+1.0),bias,nclass);
  }
  else if(params->avg == JSGD_AVG_OPT){
    float tt = t - (int)((1-params->rho)*tmax);
    if(tt>0){
      vec_scale(params->bavg,nclass,(tt-1.0)/tt);
      vec_addto(params->bavg,1.0/tt,bias,nclass);
    }
  }
  if(!wfactors){
    if(params->avg == JSGD_AVG_WGT){
      vec_scale(params->wavg,w_size,(t-1.0)/(t+1.0));
      vec_addto(params->wavg,2.0*wfactor/(t+1.0),w,w_size);
      params->nmodif += 2*nclass;
    }
    else if(params->avg == JSGD_AVG_OPT){
      float tt = t - (int)((1-params->rho)*tmax);
      if(tt>0){  
	vec_scale(params->wavg,w_size,(tt-1.0)/tt);
	vec_addto(params->wavg,wfactor/tt,w,w_size);
	params->nmodif += 2*nclass;
      }
    }
  }
  else{//OVR case
    int y;
    if(params->avg == JSGD_AVG_WGT){
      vec_scale(params->wavg,w_size,(t-1.0)/(t+1.0));
      for(y = 0; y<nclass; ++y){
	vec_addto(params->wavg+y*d,2.0*wfactors[y]/(t+1.0),w+y*d,d);
      }
      params->nmodif += 2*nclass;
    }
    else if(params->avg == JSGD_AVG_OPT){
      float tt = t - (int)((1-params->rho)*tmax);
      if(tt>0){  
	vec_scale(params->wavg,w_size,(tt-1.0)/tt);
	 for(y = 0; y<nclass; ++y){
	   vec_addto(params->wavg,wfactors[y]/tt,w,w_size);
	 }
	 params->nmodif += 2*nclass;
      }
    }
  }
}


static void swap(float* array,int* array2,int i, int j){
  float a = array[i];
  array[i] = array[j];
  array[j] = a;
  int b = array2[i];
  array2[i] = array2[j];
  array2[j] = b;
}

//Adapted from wikipedia...
static int partition(float* array,int* array2,int left, int right , int pivotIndex){
  float pivotValue = array[pivotIndex];
  swap(array,array2,pivotIndex,right-1);
  int storeIndex = left;
  int i;
  for(i = left; i<right-1; ++i){
    if(array[i] <= pivotValue){
      swap(array,array2,i,storeIndex);
      ++storeIndex;
    }
  }
  swap(array,array2,storeIndex,right-1);
  return storeIndex;
}

static void joint_quicksort(float* array, int* array2, int left, int right){
  if(right-left<2) return;
  int pivotIndex = (right+left)/2;
  int pivotNewIndex = partition(array,array2, left,right, pivotIndex);
  joint_quicksort(array,array2, left, pivotNewIndex);
  joint_quicksort(array,array2, pivotNewIndex + 1, right);
}

static void joint_sort(float* array, int* array2, int n){
  joint_quicksort(array,array2,0,n);
  
  //Check...
  int i;
  for(i=0; i<n-1; ++i){
    assert(array[i] <= array[i+1]);
  }
}

static void fine_tune_bias_ovr(const x_matrix_t *x, const int *y, int nclass, 
			       const float *w, float *bias, 
			       const jsgd_params_t *params){
  assert(params->bias_term>0);

  int n = x->n;
  float *scores = NEWA(float, nclass * n);   
  if(params->n_thread) 
    x_matrix_matmul_thread(x, w, nclass, scores);    
  else
    x_matrix_matmul(x, w, nclass, scores);

  float rho = 1.0 / (1 + params->beta); 
  int i,j;
  for(j=0; j<nclass; j++){
    float *zeros = NEWA(float,n);
    int npos=0;
    int nneg=0;
    int* labs = NEWA(int,n);
    for(i=0; i<n; ++i){
      int yil = y[i]==j? 1 : -1;
      zeros[i] = (yil - scores[i*nclass+j])/params->bias_term; //since 1/yil = yil
      if(yil>0) ++npos;
      else ++nneg;
      labs[i] = yil;
    }

    //jointly sort zeros and labs
    joint_sort(zeros,labs,n);

    //Compute slopes at each zero
    int npos_right = npos;
    int nneg_left = 0;
    float prev_slope = -1000000;
    for(i=0; i<n; ++i){
      float slope = params->bias_term*((1-rho) * nneg_left/nneg - rho * npos_right/npos);

      if(slope>=0){
	if(slope==0)//This point (or the next) is a good candidate
	  bias[j] = zeros[i];
	else{

	  bias[j] = (prev_slope*zeros[i-1]-slope*zeros[i])/(prev_slope-slope);//Interection of the two half lines.
	}
	break;
      }
      prev_slope = slope;
      if(labs[i]>0) --npos_right;
      else ++nneg_left;
    }
    free(zeros);
    free(labs);
  }
  free(scores);
}


static float eval_bias_objective(const float* scores, const x_matrix_t *x, const int *y, int nclass, const float *w, const float *bias, const jsgd_params_t *params,float regul){
  int n = x->n;
  float *current_scores = NEWA(float,nclass*n);
  memcpy(current_scores, scores, sizeof(float)*nclass*n);
  float *losses = NEWA(float,n);

  int i,j;
  for(i = 0; i < n; i++) {
    float *scores_i = current_scores + i * nclass;
    for(j = 0; j < nclass; j++) 
      scores_i[j] += bias[j] * params->bias_term;
  }
  compute_losses_with_scores(current_scores,  x, y, nclass, w, bias,params, losses);
  float loss = 0;
  for(i=0; i<n; ++i){
    loss += losses[i];
  }
  if(params->algo!= JSGD_ALGO_OVR){
    loss/=n;
  }

  free(current_scores);
  free(losses);

  return loss + regul; 
}


static int bias_descent(const float *scores, const x_matrix_t *x, const int *y, int nclass, float *bias, const jsgd_params_t *params, long t, float eta0){
  float eta = eta0/(1+t*eta0);
  int i = t%x->n,j;
  int ybar=-1;
  switch(params->algo){
  case JSGD_ALGO_MUL: case JSGD_ALGO_MUL2:
    {
      float best_score = -1e20;    
      for(j = 0; j < nclass; j++) {
	float score = scores[i*nclass + j] + bias[j] * params->bias_term;
	if(fabs(score)>1e20) //problem with eta0-> leads to too high values
	  return 1;
	if(j != y[i]) score += 1;
	if(score > best_score) {
	  best_score = score; 
	  ybar = j;       
	}
      }
      float factor=1;
      if(params->algo==JSGD_ALGO_MUL2){
	factor = 2*(scores[i*nclass+ybar]-scores[i*nclass+y[i]] + params->bias_term*(bias[ybar]-bias[y[i]]) + 1);
      }
      bias[ybar] -= eta*params->bias_term*factor;
      bias[y[i]] += eta*params->bias_term*factor;
    }
    break;
  case JSGD_ALGO_LOG: case JSGD_ALGO_STF:
    {
      float temp = params->algo==JSGD_ALGO_STF ? params->temperature : 1;
      int delta = params->algo==JSGD_ALGO_STF;
      float best_score = -1e20;    
      float* p = NEWA(float,nclass);
      float* bgrad = NEWA(float,nclass);
      for(j=0;j<nclass; ++j){
	p[j] = temp * (scores[i*nclass + j] + params->bias_term * bias[j] + delta * (j==y[i] ? 0 : 1));
      }

      for(j = 0; j < nclass; j++) {
	if(p[j] > best_score) {
	  best_score = p[j];       
	}
      }
      float sum=0;
      for(j = 0; j < nclass; ++j) {
	p[j] = exp(p[j] - best_score);
	sum += p[j];
      }

      for(j=0; j<nclass; ++j){
	bgrad[j] = params->bias_term * p[j]/sum;
      }
      bgrad[y[i]] -= params->bias_term;

      for(j=0; j<nclass; ++j){
	bias[j] -= eta*bgrad[j];
      }
      free(p);
      free(bgrad);
    }
    break;
  default:
    assert(!"Bias tuning unimplemented, continuing.\n");
  }
  return 0;
}

static void fine_tune_bias_others(const x_matrix_t *x, const int *y, int nclass, 
				  const float *w, float *bias, 
				  const jsgd_params_t *params){

  int n = x->n, d=x->d;

  float *scores = NEWA(float, nclass * n);   
  if(params->n_thread) 
    x_matrix_matmul_thread(x, w, nclass, scores);    
  else
    x_matrix_matmul(x, w, nclass, scores);

  float regul = params->lambda/2 * vec_sqnorm(w,nclass*d);

  //Determine eta0
  int netas = params->biasopt_netas;//Has to be odd
  float etaf = params->biasopt_etafactor;
  int nsample_eta = params->biasopt_nsample_eta;
  float etas[netas];
  int i;
  etas[netas/2]=1;
  for(i = 1; i<=netas/2; ++i){
    etas[netas/2+i] = etas[netas/2+i-1]*etaf;
    etas[netas/2-i] = etas[netas/2-i+1]/etaf;
  }
  /*for(i=0;i<netas;++i){
    printf("%g ",etas[i]);
  }
  printf("\n");*/
  int id_eta;
  float* b = NEWA(float,nclass);

  float best_obj=10000000;
  float best_eta=etas[0];
  for(id_eta = 0; id_eta<netas; ++id_eta){
    memcpy(b,bias,sizeof(float)*nclass);
    int error=0;
    for(i=0; i<nsample_eta; ++i){
      error = bias_descent(scores,x,y,nclass,b,params,i,etas[id_eta]);
      if(error) break;
    }
    if(error){
      if(params->verbose) printf("Bias eta=%g led to too large values,skipping \n",etas[id_eta]);
      continue;
    }
    float obj =  eval_bias_objective(scores,x,y,nclass,w,b,params,regul);
    //printf("%g\n",obj);
    if(obj<best_obj){
      best_eta = etas[id_eta];
      best_obj = obj;
    }
  }
  //printf("selected eta: %g with objective %g\n",best_eta,best_obj);
  
  //Launch real b optimization
  const int b_nepochs = params->biasopt_n_epoch;
  int t;
  for(t=0; t<n*b_nepochs; ++t){
    /*if(t%n==0){
      float obj = eval_bias_objective(scores,x,y,nclass,w,bias,params,regul);
      printf("Bias obj: %g\n",obj);
      }*/
    bias_descent(scores,x,y,nclass,bias,params,t,best_eta);
  }
  

  free(b);
  free(scores);
}


static void fine_tune_bias(const x_matrix_t *x, const int *y, int nclass, 
			   const float *w, float *bias, 
			   const jsgd_params_t *params){
  float initial_objective = compute_cost(x,y,nclass,w,bias,params);

  printf("Bias post-optimization...\n");

  switch(params->algo){
  case JSGD_ALGO_OVR:
    fine_tune_bias_ovr(x,y,nclass,w,bias,params);
    break;
  case JSGD_ALGO_MUL:case JSGD_ALGO_MUL2: case JSGD_ALGO_LOG: case JSGD_ALGO_STF:
    fine_tune_bias_others(x,y,nclass,w,bias,params);
    break;
  default:
    printf("Warning: bias fine-tuning not implemented, continuing...");
  }
 
  float final_objective = compute_cost(x,y,nclass,w,bias,params);
  printf("Bias improvement: %g ----> %g\n",initial_objective,final_objective);
 
}
