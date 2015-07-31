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

class LearnModels
{
public:
	LearnModels();
	void loadTrainingData();
	void computeFeaturesDocs();
	void getImagesDocs();
	void loadModels();
	void learnModel(const Doc& doc, const CharInstance& ci, HogSvmModel &hs_model);

	/* For debugging */
	void train();
	//void getSvmDetector(uchar asciiCode, Mat &sv, double &rho);
	//void computeScoresL2(const Mat & HogFeatures, const Mat & w, double rho, int bH, int bW, int dim, 
	//	int nbinsH, int nbinsW, int step, Mat & score);
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

	// TODO: unite to a single map.
	unordered_map<uchar, Size> m_WindowSz;
	unordered_map<uchar, uint> m_classes;

	LMParams m_params;
};
#endif // !_H_LEARN_MODELS_H__
