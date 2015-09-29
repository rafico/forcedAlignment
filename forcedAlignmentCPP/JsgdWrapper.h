#ifndef _H__JSGD_WRAPPER_H__
#define _H__JSGD_WRAPPER_H__

// This library doesn't compile under windows.
#ifndef _WIN32

#include "commonTypes.h"
#include "jsgd.h"


class JsgdWrapper
{
public:
	static void trainModel(Mat labels, Mat trainingData, vector<float>& weight);
};

#endif // !_WIN32
#endif // !_H__JSGD_WRAPPER_H__