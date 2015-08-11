#ifndef _H__JSGD_WRAPPER_H__
#define _H__JSGD_WRAPPER_H__

#include "commonTypes.h"
#include "jsgd.h"

class JsgdWrapper
{
public:
	JsgdWrapper();
	void trainModel(Mat labels, Mat trainingData, vector<float>& weight);

private:
	jsgd_params_t m_params;
};


#endif // !_H__JSGD_WRAPPER_H__