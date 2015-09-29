#ifndef _H_CHAR_SPOTTING_H__
#define _H_CHAR_SPOTTING_H__

#include "commonTypes.h"
#include "CharClassifier.h"
#include "TrainingData.h"

class CharSpotting
{
public:
	CharSpotting();
	
	void evaluateModels(bool ExemplarModel=false);

private:
	void loadData();

	const vector<Rect>& getRelevantBoxesByClass(uint classNum, uint docNum) { return m_relevantBoxesByClass[docNum*m_numClasses + classNum]; }
	void evalModel(const HogSvmModel& hs_model, uint classNum, vector<double> &scores, vector<double> &resultLabels, vector<pair<Rect, size_t>> & locWords);
	void saveResultImages(const Doc& doc, const Character& query, const vector<double>& resultLabels, const vector<pair<Rect, size_t>>& locWords);
	void compute_mAP(const vector<double> &resultLabels, uint nrelW, double &mAP, double &rec);

	vector<uint> m_numRelevantWordsByClass;
	vector<vector<Rect>> m_relevantBoxesByClass;
	size_t m_numClasses;
	unordered_map<uchar, uint> m_classes;

	const vector<Doc>& m_trainingDocs;
	const Params &m_params;

	CharClassifier m_classifier;
};

#endif // !_H_LEARN_MODELS_H__