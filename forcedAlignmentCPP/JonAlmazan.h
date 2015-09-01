#ifndef _H_JON_ALMAZAN_H__
#define _H_JON_ALMAZAN_H__

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect.hpp>
#include "commonTypes.h"
#include "LMParams.h"
#include "CharInstance.h"
#include "Doc.h"
#include "HogSvmModel.h"

class JonAlmazan
{
public:
	JonAlmazan();
	void loadTrainingData();
	void computeFeaturesDocs();
	void getImagesDocs();
	void LearnModelsAndEvaluate();
	void learnModel(const CharInstance& query, HogSvmModel &hs_model);
	void evalModel(const HogSvmModel& hs_model, uint classNum, vector<double> &scores, vector<double> &resultLabels, vector<pair<Rect, size_t>> & locWords);
	void getWindows(const Doc& doc, const HogSvmModel& hs_model, vector<double>& scsW, vector<Rect>& locW);
	vector<int> nms(Mat I, const vector<Rect>& X, double overlap);
	const vector<Rect>& getRelevantBoxesByClass(uint classNum, uint docNum) { return m_relevantBoxesByClass[docNum*m_numClasses + classNum]; }
	void saveResultImages(const Doc& doc, const CharInstance& query, const vector<double>& resultLabels, const vector<pair<Rect, size_t>>& locWords);
	void compute_mAP(const vector<double> &resultLabels, uint nrelW, double &mAP, double &rec);

	/* For debugging */
	void train();
	//void getSvmDetector(uchar asciiCode, Mat &sv, double &rho);
	//Size getWindowSz(uchar asciiCode) { return m_WindowSz.at(asciiCode); };
	//Size getHOGWindowSz(uchar asciiCode); 

private:
	void NormalizeFeatures(Mat & features);
	//void drawLocations(Mat &img, const vector<Rect> &locations, const Scalar &color);
	//void samplePos(Mat & posLst, const Size & size, const vector<ModelInstance>& miVec);
	//void sampleNeg(uchar asciiCode, cv::Mat& features, size_t startPos, const Mat & negHogFeatures, Size imageHogSz, Size modelHogSz, size_t numOfExamples);
	//void computeAverageWindowSize4Char();
	//void trainSvm(Ptr<SVM>, const Mat & trainData, const vector<int> & labels, uchar asciiCode);

	vector<uint> m_numRelevantWordsByClass;
	vector<vector<Rect>> m_relevantBoxesByClass;
	vector<Doc> m_docs;
	size_t m_numClasses;

	// TODO: unite to a single map.
	unordered_map<uchar, Size> m_WindowSz;
	unordered_map<uchar, uint> m_classes;

	LMParams m_params;

};
#endif // !_H_JON_ALMAZAN_H__
