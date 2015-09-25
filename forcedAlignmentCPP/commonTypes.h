#ifndef _H_COMMON_TYPES_H__
#define _H_COMMON_TYPES_H__

#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>

#define MISPAR_KATAN_MEOD (-1000000)
#define MAX_LINE_SIZE 4096

using std::string;
using std::vector;
using std::unordered_map;
using std::pair;
using std::cout;
using std::endl;

using cv::Rect;
using cv::Size;
using cv::Mat;
using cv::Scalar;
using cv::Ptr;
using cv::ml::SVM;
using cv::Range;

typedef unsigned char uchar;
typedef unsigned int uint;

class StringVector : public std::vector<std::string> {
public:

	unsigned int read(const std::string &filename) {
		std::ifstream ifs;
		char line[MAX_LINE_SIZE];
		ifs.open(filename.c_str());
		if (!ifs.is_open()) {
			std::cerr << "Unable to open file list:" << filename << std::endl;
			return 0;
		}
		while (!ifs.eof()) {
			ifs.getline(line, MAX_LINE_SIZE);
			if (strcmp(line, ""))
				push_back(std::string(line));
		}
		ifs.close();
		return size();
	}
};

#endif // !_H_COMMON_TYPES_H__


