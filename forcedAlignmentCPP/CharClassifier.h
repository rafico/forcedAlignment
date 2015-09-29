#ifndef _H_LEARN_MODELS_H__
#define _H_LEARN_MODELS_H__

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect.hpp>
#include "commonTypes.h"
#include "Params.h"
#include "Doc.h"
#include "HogSvmModel.h"
#include "TrainingData.h"

class CharClassifier
{
public:
	CharClassifier();
	void computeFeaturesDocs();
	void learnModels();

	HogSvmModel learnModel(uchar asciiCode);
	HogSvmModel learnExemplarModel(const Doc& doc, const Character& query);
	
	void load_char_stats(charStatType &meanCont, charStatType& stdCont) const;
	void getMinMaxCharLength(uint &maxCharLen, uint &minCharLen);

	vector<Doc> &getTrainingDocs() { return m_trData.getTrainingDocs(); }
	const Doc &getDocByName(const string &docName) { return m_trData.getDocByName(docName); }

private:
	void NormalizeFeatures(Mat & features);
	void samplePos(const Mat &imDoc, Mat &trHOGs, size_t &position, const Rect &loc, Size sz);
	void sampleNeg(Mat &trHOGs, size_t position, int wordsByDoc, const HogSvmModel &hs_model);
	void trainClassifier(const Mat &trHOGs, HogSvmModel &hs_model, size_t numSamples, size_t numTrWords);

	Params& m_params;

	unordered_map<uchar, HogSvmModel> m_svmModels;

	TrainingData m_trData;
	vector<Doc> &m_docs;
};
#endif // !_H_LEARN_MODELS_H__
