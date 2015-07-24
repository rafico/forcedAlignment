#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect.hpp>
#include "commonTypes.h"
#include "LMParams.h"
#include "query.h"

/*
each 'csv' file represents a matrix of size numOfCharacter x 6, and each row in the following format

Threshold  Ascii_code X  Y W H

(X, Y) is the top left coordinate of the character's bounding box
W, H - are the width and height coordinates of the bounding box
*/
struct ModelInstance
{
	ModelInstance(const string &fileName, const string &csv_line);

	string m_fileName;

	uchar m_threshold;
	uchar m_asciiCode;
	Rect m_window;
};

class LearnModels
{
public:
	LearnModels(const std::string& models_path = "", const std::string& images_path = "");
	void getDocs();
	void getQueries();

	/* For debugging */
	void train();
	void getSvmDetector(uchar asciiCode, Mat &sv, double &rho);
	void computeScoresL2(const Mat & HogFeatures, const Mat & w, double rho, int bH, int bW, int dim, 
		int nbinsH, int nbinsW, int step, Mat & score);
	Size getWindowSz(uchar asciiCode) { return m_WindowSz.at(asciiCode); };
	Size getHOGWindowSz(uchar asciiCode); 
	~LearnModels();

private:
	void drawLocations(Mat &img, const vector<Rect> &locations, const Scalar &color);
	void samplePos(Mat & posLst, const Size & size, const vector<ModelInstance>& miVec);
	void sampleNeg(uchar asciiCode, cv::Mat& features, size_t startPos, const Mat & negHogFeatures, Size imageHogSz, Size modelHogSz, size_t numOfExamples);
	void computeAverageWindowSize4Char();
	void NormalizeFeatures(Mat & features);
	void trainSvm(cv::Ptr<SVM>, const cv::Mat & trainData, const vector<int> & labels, uchar asciiCode);
	
	vector<Query> m_queries;

	/* Associates file name to the relevant image. */
	unordered_map<string, Mat> m_trainingImages;
	
	// TODO: unite to a single map.
	unordered_map<uchar, vector<ModelInstance>> m_modelInstances;
	unordered_map<uchar, Ptr<SVM>> m_SVMModels;
	unordered_map<uchar, Size> m_WindowSz;

	LMParams m_params;
};