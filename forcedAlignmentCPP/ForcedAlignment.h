#ifndef _H_FORCED_ALIGNMENT_TRAIN_H__
#define _H_FORCED_ALIGNMENT_TRAIN_H__

#include "Dataset.h"

class ForcedAlignment
{
public:
	ForcedAlignment();

	void train();
	void decode();

private:
	void drawSegResult(const AnnotatedLine &x, const StartTimeSequence &y, const StartTimeSequence &y_hat);
	void drawSegResult(const AnnotatedLine &x, const StartTimeSequence &y, string resultPath);

	const string m_classifier_filename;
};


#endif // !_H_FORCED_ALIGNMENT_TRAIN_H__