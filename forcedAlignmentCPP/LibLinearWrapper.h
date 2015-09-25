#ifndef _H__LIB_LINEAR_WRAPPER_H__
#define _H__LIB_LINEAR_WRAPPER_H__

#include "commonTypes.h"
#include "linear.h"

class LibLinearWrapper
{
public:
	LibLinearWrapper();
	void trainModel(Mat labels, Mat trainingData, vector<float>& weight);

private:
	struct parameter m_param;
	struct problem m_prob;
};


#endif // !_H__LIB_LINEAR_WRAPPER_H__
