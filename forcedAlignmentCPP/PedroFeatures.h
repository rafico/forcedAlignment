#ifndef _H_PEDRO_FEATURES_H__
#define _H_PEDRO_FEATURES_H__

#include <opencv2/core.hpp>

class PedroFeatures
{
public:
	PedroFeatures() = delete;
	static cv::Mat process(cv::Mat image, int sbin, int *h = 0, int *w = 0);
};


#endif // !_H_PEDRO_FEATURES_H__
