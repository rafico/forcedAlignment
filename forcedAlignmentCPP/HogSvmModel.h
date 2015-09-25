#ifndef _H_HOG_SVM_MODEL_H__
#define _H_HOG_SVM_MODEL_H__

#include "commonTypes.h"

struct HogSvmModel
{
	HogSvmModel() {};
	HogSvmModel(uchar asciiCode, const string& pathCharModels = "");

	void save2File();
	bool loadFromFile();

	uchar m_asciiCode;

	int m_newH;
	int m_newW;
	int m_bH;
	int m_bW;
	vector<float> weight;
	double m_bias;

	string m_fileName;

};


#endif // !_H_HOG_SVM_MODEL_H__
