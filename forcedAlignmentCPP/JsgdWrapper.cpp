#ifndef _WIN32 

#include "JsgdWrapper.h"

void JsgdWrapper::trainModel(Mat labels, Mat trainingData, vector<float>& weight)
{
	CV_Assert(trainingData.type() == CV_32F);
	Mat labels_int;
	labels.convertTo(labels_int, CV_32S);
	    
  	/* training matrix */
  	x_matrix_t x; 
	x.encoding = x_matrix_t::JSGD_X_FULL;
	x.n = trainingData.rows;
	x.d = trainingData.cols;
	x.data = trainingData.ptr<float>(0);
   
	/* algorithm parameters */

 	jsgd_params_t params;
 	jsgd_params_set_default(&params); 

	params.algo = jsgd_params_t::JSGD_ALGO_OVR;
	params.lambda = 0.00001;
	params.bias_term = 1;
	params.eta0 = 0.001;
	params.verbose = 2;
	params.eval_freq = params.n_epoch = 10;
	params.beta = 1;

	int nclass = 2;
	int *labels_0 = labels.ptr<int>(0);
	float bias[nclass];

	vector<float> w(x.d*2);

	/* the call */
	jsgd_train(nclass, &x, labels_0, w.data(), bias, &params);
	weight.assign(w.begin(),w.begin()+x.d);
}
#endif // !_WIN32
