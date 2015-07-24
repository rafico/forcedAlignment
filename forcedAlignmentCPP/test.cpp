#include <chrono>
#include <iostream>
#include "LearnModels.h"
#include "PedroFeatures.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	/* testLineExtraction(); */
	//TODO: clean up the dataset of character, it contains noise.

	LearnModels lm;
	lm.train();

	Mat line = imread("D:/Dropbox/Code/forcedAlignment/croppedLine.png");
	int bH, bW;
	cv::Mat features = PedroFeatures::process(line, 8, &bH, &bW);
	Mat w;
	double rho;
	lm.getSvmDetector('b', w, rho);
	Size sz = lm.getHOGWindowSz('b');
	Mat score(bH - sz.height + 1, bW - sz.width + 1, CV_64F);
	features.convertTo(features, CV_32F);
	lm.computeScoresL2(features, w, rho, bH, bW, 31, sz.height, sz.width, 1, score);

	double minVal, maxVal;
	int minIdx, maxIdx;
	minMaxIdx(score, &minVal, &maxVal, &minIdx, &maxIdx);


	cout << "Press any key to continue.";
	auto c = getchar();
	return 0;
}