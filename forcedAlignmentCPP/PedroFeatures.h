#include <opencv2/core.hpp>

class PedroFeatures
{
public:
	PedroFeatures() = delete;
	static cv::Mat process(cv::Mat image, int sbin, int *h = 0, int *w = 0);
};

