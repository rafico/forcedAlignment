#ifndef _H_CLASSIFIER_H__
#define _H_CLASSIFIER_H__

#include "commonTypes.h"
#include "Doc.h" 

class Classifier
{
public:
	void loadLine(const string& lineFileName);

private:
	vector<Doc> m_lines;
};

#endif // ! _H_CLASSIFIER_H__