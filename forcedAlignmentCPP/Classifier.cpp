#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "Classifier.h"
#include "LearnModels.h"
#include "HogSvmModel.h"
#include "HogUtils.h"

using namespace cv;
using namespace std;

void Classifier::loadLine(const string& lineFileName)
{
	/*
	uint sbin = 6;

	Doc doc(lineFileName);
	doc.resizeDoc(sbin);
	doc.computeFeatures(sbin);

	LearnModels lm;
	lm.learnModels();

	HogSvmModel hs_model = lm.learnModel('b');

	vector<Rect> locW;
	vector<double> scsW;
	HogUtils::getWindows(doc, hs_model, scsW, locW, 1, sbin);

	Mat I;
	sortIdx(Mat(scsW), I, SORT_EVERY_COLUMN + CV_SORT_DESCENDING);

	for (int i = 0; i < 15; ++i)
	{
		auto index = I.at<int>(i);
		rectangle(doc.m_origImage, locW[index],	Scalar(0, 0, 255));
	}
	
	Mat test;
	Size sz = Size(round(doc.m_image.cols / 2), round(doc.m_image.rows / 2));
	pyrDown(doc.m_image, test, sz);

	namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Display window", doc.m_origImage);

	waitKey(0);

	int x = 2;
	*/
}
