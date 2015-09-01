#ifndef _H_HOG_SVM_MODEL_H__
#define _H_HOG_SVM_MODEL_H__

#include "commonTypes.h"

struct HogSvmModel
{
	HogSvmModel(const string& pathCharModels, uchar asciiCode);

	void save2File();
	bool loadFromFile();

	int m_newH;
	int m_newW;
	int m_bH;
	int m_bW;
	vector<float> weight;
	double m_bias;

	string m_fileName;

};


#endif // !_H_HOG_SVM_MODEL_H__
