#ifndef SGD_H_INCLUDED
#define SGD_H_INCLUDED

#include "x_matrix.h"



/* Type of a callback function to compute the accuracy of a score. */
typedef double jsgd_accuracy_function_t(
      const float *scores,  /* size (nclass, n) classification score for each example for each class */
      const int *y,         /* size (n) label for each example */
      int nclass,           /* nb of classes */
      int n,                /* nb of examples */
      void *arg);           /* arbitrary pointer that can be used as context */



/* Parameters, statistics and temp data for the learning
 * algorithm. Most values can be kept at 0, call
 * jsgd_params_set_default for the remaining ones */

typedef struct {
  int verbose;     /* verbosity level */
  int n_thread;    /* number of threads to use (0 = never create threads) */
  int random_seed; /* running with the same random seed should give the same results */

  /* initialization */
  int use_input_w; /* use input w as initialization (else it is cleared) */
  
  enum {
    JSGD_ALGO_OVR, /*One versus rest*/
    JSGD_ALGO_MUL, /*multiclass svm*/
    JSGD_ALGO_MUL2, /*Squared hinge loss*/
    JSGD_ALGO_RNK, /*ranking */
    JSGD_ALGO_WAR, /*weighted average ranking */
    JSGD_ALGO_LOG, /*Multinomial logistic regression */
    JSGD_ALGO_STF, /*Stiff logistic loss */
    JSGD_ALGO_SQR, /* Multiclass Square loss */
  } algo;

  long n_epoch;            /* number of samples visited */
  long eta_nepoch;          /* Number of epochs for the determination of eta0*/

  /* validation set to decide stopping + report stats */
  x_matrix_t *valid;        /* validation matrix */
  int *valid_labels;        /* corresponding labels (0 based) */

  /* evaluation passes */

  int eval_freq;               /* evaluate on validation set every that many epochs */
  int compute_train_accuracy;  /* (boolean) disabled by default, may be expensive */
  double stop_valid_threshold; /* stop if the accuracy is lower than
                                  the accuracy at the previous
                                  evaluation + this much */
  jsgd_accuracy_function_t *accuracy_function;   
                               /* pointer to an alternative function
                                  to compute the accuracy */
  void * accuracy_function_arg; /* arbitrary pointer passed as arg to
                                   the accuracy function */ 
  
  enum {
    JSGD_AVG_NONE, /*No averaging (returns final w)*/
    JSGD_AVG_WGT,  /*weighted average (returns  2/(T*(T+1)) sum t*w)*/
    JSGD_AVG_OPT,  /*Optimal averaging (returns sum w over the last alpha*T */
  } avg;

  float rho; /* In (0,1], proportion of last weights to take into account for averaging. */
  
  /* algorithm parameters */
  double lambda;            
  double bias_term;         /* this term is appended to each training vector */
  double eta0;              /* intial value of the gradient step */
  int beta;                 /* for OVR, how many negatives to sample per training vector */
  int fixed_eta;            /* (boolean) use a fixed step */
  float temperature;         /* inverse temperature parameter for Stiff loss */
  int fine_tune_bias;       /* If true, reoptimizes in the end to find the optimal bias */
  float bias_learning_rate;  /* Multiplies eta by this value to update bias */

  /* Algorithm paramters for bias post-optimization (should not need to be changed) */
  int biasopt_netas;           /* Number of chosen etas (default 9). Should be odd. */
  double biasopt_etafactor;    /* Scale factor of two possible consecutive etas. (default 10)*/
  int biasopt_nsample_eta;     /* Maximum number of feature points kept for the eta determination (default 10000)*/
  long biasopt_n_epoch;        /* Number of epochs (default 100) */
			      


  /* output stats */  
  int best_epoch;           /* epoch at which the result was found */
  long niter;               /* effective number of iterations */
  long ndp, nmodif;         /* nb of dot products performed, number of W column modifications */
  double t_eval;            /* time spent on evaluation passes */
  int eval_objective;       /* evaluate full objective every epoch */
  float* objectives;        /* output objective array */

  /* stats stampled at each evaluation */
  long na_stat_tables;      /* sizes of following tables (allocated by caller) */
  double *times;            /* timestamp at each evaluation (seconds) */
  double *train_accuracies; 
  double *valid_accuracies; 
  int *ndotprods, *nmodifs; 


  /* internal parameters and temp data */
  double best_valid_accuracy;   /* best validation accuracy seen so far */
  double t0;                    /* begin time */
  unsigned int rand_state;      /* current random state */
  int t_block, n_wstep, d_step; /* blocking parameters */
  float *self_dotprods;         /* buffer for dot products between the x(:, i)'s */
  int use_self_dotprods;        /* should we use them? */
  int check_level;              /* 0: no check (default), 1: check loss, 2: check_gradient, 3: 1+2 */
  float *wavg;                  /* Used to store the averaged result */
  float *bavg;                  /* Averaged bias */

} jsgd_params_t; 



/* fill in decent default params (OVR) */
void jsgd_params_set_default(jsgd_params_t *params); 

/* Uses SGD to learn a classifier
 * labels should be in 0:nclass-1
 * 
 * @param nclass       nb of classes
 * @param x            matrix of train vectors, size (d, n)
 * @param y(n)         labels of each vector. Should be in 0:nclass-1
 * @param w(d,nclass)  output w matrix
 * @param bias(nclass) output bias coefficients
 * @param params       all other parameters 
 *
 * @return             nb of iterations performed
 */

int jsgd_train(int nclass,
               const x_matrix_t *x,
               const int *y,
               float * w, 
               float * bias, 
               jsgd_params_t *params); 


/* Compute classification scores for a matrix. 
 * 
 * @param nclass           nb of classes
 * @param x                matrix of vectors to classifiy, size (d, n)
 * @param w(d,nclass)      w matrix
 * @param bias(nclass)     bias coefficients
 * @param threaded         (boolean) whether to use threads
 * @param scores(nclass,n) output classification scores
 */

void jsgd_compute_scores(const x_matrix_t *x, int nclass, 
                         const float *w, const float *bias, 
                         float bias_term, 
                         int threaded, 
                         float *scores);

#endif
