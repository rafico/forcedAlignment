#include <string>
#include <unordered_map>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect.hpp>

typedef unsigned char uchar;
typedef unsigned int uint;

/*
each 'csv' file represents a matrix of size numOfCharacter x 6, and each row in the following format

Threshold  Ascii_code X  Y W H

(X, Y) is the top left coordinate of the character's bounding box
W, H - are the width and height coordinates of the bounding box
*/
struct ModelInstance
{
	ModelInstance(const std::string &fileName, const std::string &csv_line);

	std::string m_fileName;

	uchar m_threshold;
	uchar m_asciiCode;
	cv::Rect m_window;
};

struct LMParams
{
	LMParams(uint sbin = 8, cv::Range rangeX = cv::Range(-10, 12), cv::Range rangeY = cv::Range(-10, 12), uint stepSize = 4, uint propNWords = 2)
		: m_sbin(sbin),
		m_rangeX(rangeX),
		m_rangeY(rangeY),
		m_stepSize4PositiveExamples(stepSize),
		m_propNWords(propNWords),
		m_numTrWords((rangeX.size() / m_stepSize4PositiveExamples) * (rangeY.size() / m_stepSize4PositiveExamples)),
		m_svmModelsLocation("D:/Dropbox/Code/forcedAlignmentCPP/svmModels/")
	{}

	//cell size.
	uint m_sbin;
	cv::Range m_rangeX;
	cv::Range m_rangeY;
	uint m_stepSize4PositiveExamples;
	uint m_propNWords;
	uint m_numTrWords;
	std::string m_svmModelsLocation;
};

class LearnModels
{
public:
	LearnModels(const std::string& models_path = "", const std::string& images_path = "");
	/* For debugging */
	void train();

	~LearnModels();

private:
	void getSvmDetector(const cv::Ptr<cv::ml::SVM>& svm, std::vector<float> & hog_detector);
	void drawLocations(cv::Mat &img, const std::vector<cv::Rect> &locations, const cv::Scalar &color);
	void samplePos(cv::Mat & posLst, const cv::Size & size, const std::vector<ModelInstance>& miVec);
	void sampleNeg(cv::Mat& features, size_t startPos, const cv::Mat & negHogFeatures, cv::Size imageHogSz, cv::Size modelHogSz, size_t numOfExamples);
	cv::Size computeAverageWindowSize4Char(const std::vector<ModelInstance>& mi_vec);
	void NormalizeFeatures(cv::Mat & features);
	void trainSvm(cv::Ptr<cv::ml::SVM>, const cv::Mat & trainData, const std::vector<int> & labels, uchar asciiCode);

	/* Associates file name to the relevant image. */
	std::unordered_map<std::string, cv::Mat> m_trainingImages;
	std::unordered_map<uchar, std::vector<ModelInstance>> m_modelInstances;
	std::unordered_map<uchar, cv::Ptr<cv::ml::SVM>> m_SVMModels;

	cv::Ptr<cv::ml::SVM> m_svm;
	LMParams m_params;
};