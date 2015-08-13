#ifndef _H__JSGD_WRAPPER_H__
#define _H__JSGD_WRAPPER_H__

#include "commonTypes.h"

// This library doesn't compile under windows and I don't have the time to port it.
#ifndef _WIN32 
	#include "jsgd.h"
#endif // !_WIN32

class JsgdWrapper
{
public:
	static void trainModel(Mat labels, Mat trainingData, vector<float>& weight);
};


#endif // !_H__JSGD_WRAPPER_H__