#include "JsgdWrapper.h"

JsgdWrapper::JsgdWrapper()
{
	jsgd_params_set_default(&m_params);	  
}

void JsgdWrapper::trainModel(Mat labels, Mat trainingData, vector<float>& weight)
{
	/* training matrix */
	x_matrix_t x; 
	x.encoding = x_matrix_t::JSGD_X_FULL;
	/*
	x->n = mxGetN (a);
	x->d = mxGetM (a);
	x->data = mxGetData(a); 
	*/
}