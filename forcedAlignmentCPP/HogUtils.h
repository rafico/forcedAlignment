#ifndef _H_HOG_UTILS_H__
#define _H_HOG_UTILS_H__

#include <opencv2/core.hpp>
#include "commonTypes.h"
#include "Doc.h"
#include "HogSvmModel.h"

class HogUtils
{
public:
	HogUtils() = delete;

	static Mat process(Mat image, int sbin, int *h = 0, int *w = 0);
	static void getWindows(const Doc& doc, const HogSvmModel& hs_model, vector<double>& scsW, vector<Rect>& locW, uint step, uint sbin);
	static vector<int> HogUtils::nms(Mat I, const vector<Rect>& X, double overlap);
};


#endif // !_H_HOG_UTILS_H__
