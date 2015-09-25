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
	void loadTrainingData();
	void computeFeaturesDocs();
	void learnModels();

	void evaluateModels(bool ExemplarModel = true);
	HogSvmModel learnModel(uchar asciiCode);
	HogSvmModel learnExemplarModel(const Doc& doc, const Character& query);
	
	void evalModel(const HogSvmModel& hs_model, uint classNum, vector<double> &scores, vector<double> &resultLabels, vector<pair<Rect, size_t>> & locWords);
	const vector<Rect>& getRelevantBoxesByClass(uint classNum, uint docNum) { return m_relevantBoxesByClass[docNum*m_numClasses + classNum];}
	void saveResultImages(const Doc& doc, const Character& query, const vector<double>& resultLabels, const vector<pair<Rect, size_t>>& locWords);
	void compute_mAP(const vector<double> &resultLabels, uint nrelW, double &mAP, double &rec);

	void load_char_stats(charStatType &meanCont, charStatType& stdCont) const;
	void getMinMaxCharLength(uint &maxCharLen, uint &minCharLen);

private:
	void NormalizeFeatures(Mat & features);
	void samplePos(const Mat &imDoc, Mat &trHOGs, size_t &position, const Rect &loc, Size sz);
	void sampleNeg(Mat &trHOGs, size_t position, int wordsByDoc, const HogSvmModel &hs_model);
	void trainClassifier(const Mat &trHOGs, HogSvmModel &hs_model, size_t numSamples, size_t numTrWords);

	Params& m_params;

	vector<uint> m_numRelevantWordsByClass;
	vector<vector<Rect>> m_relevantBoxesByClass;
	vector<Doc> m_docs;
	size_t m_numClasses;

	unordered_map<uchar, uint> m_classes;
	unordered_map<uchar, HogSvmModel> m_svmModels;

	TrainingData m_trData;
};
#endif // !_H_LEARN_MODELS_H__
