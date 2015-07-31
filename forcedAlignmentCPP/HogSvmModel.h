#ifndef _H_HOG_SVM_MODEL_H__
#define _H_HOG_SVM_MODEL_H__

#include "commonTypes.h"

struct HogSvmModel
{
	uint m_newH;
	uint m_newW;
	uint m_bH;
	uint m_bW;
	vector<double> weight;
	double m_bias;
};


#endif // !_H_HOG_SVM_MODEL_H__
