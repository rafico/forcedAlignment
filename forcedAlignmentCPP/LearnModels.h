#ifndef _H_LEARN_MODELS_H__
#define _H_LEARN_MODELS_H__

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect.hpp>
#include "commonTypes.h"
#include "LMParams.h"
#include "CharInstance.h"
#include "Doc.h"
#include "HogSvmModel.h"
#include "TrainingChars.h"

class LearnModels
{
public:
	LearnModels();
	void loadTrainingData();
	void computeFeaturesDocs();
	void getImagesDocs();
	void learnModels();
	void evaluateModels();
	void learnModel(uchar asciiCode, HogSvmModel &hs_model);
	void evalModel(const HogSvmModel& hs_model, uint classNum, vector<double> &scores, vector<double> &resultLabels, vector<pair<Rect, size_t>> & locWords);
	void getWindows(const Doc& doc, const HogSvmModel& hs_model, vector<double>& scsW, vector<Rect>& locW);
	vector<int> nms(Mat I, const vector<Rect>& X, double overlap);
	const vector<Rect>& getRelevantBoxesByClass(uint classNum, uint docNum) { return m_relevantBoxesByClass[docNum*m_numClasses + classNum];}
	void saveResultImages(const Doc& doc, const CharInstance& query, const vector<double>& resultLabels, const vector<pair<Rect, size_t>>& locWords);
	void compute_mAP(const vector<double> &resultLabels, uint nrelW, double &mAP, double &rec);

	/* For debugging */
	void train();

private:
	void NormalizeFeatures(Mat & features);

	vector<uint> m_numRelevantWordsByClass;
	vector<vector<Rect>> m_relevantBoxesByClass;
	vector<Doc> m_docs;
	size_t m_numClasses;

	unordered_map<uchar, uint> m_classes;
	unordered_map<uchar, HogSvmModel> m_svmModels;

	LMParams m_params;
	TrainingChars m_trChs;
};
#endif // !_H_LEARN_MODELS_H__
