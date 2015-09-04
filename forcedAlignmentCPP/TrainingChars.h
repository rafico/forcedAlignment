#ifndef _H_TRAINING_CHARS_H__
#define _H_TRAINING_CHARS_H__

#include "commonTypes.h"
#include "CharInstance.h"
#include "Params.h"

//TODO: equip with an iterator.

struct TrainingCharsHelper
{
	double m_widthMean;
	double m_widthVariance;
	double m_heightMean;
	double m_heightVariance;
	vector<CharInstance> m_instances;
};

using TrainingCharsCont = unordered_map<uchar, TrainingCharsHelper>;

struct TrainingChars
{
	TrainingChars(const Params& params);

	void addCharInstance(const CharInstance& ci);
	const vector<CharInstance>& getSamples(uint asciiCode);
	double getMeanWidth(uint asciiCode);
	double getMeanHeight(uint asciiCode);

	void computeNormalDistributionParams();

	TrainingCharsCont m_charInstances;
	const Params& m_params;
};

#endif // !_H_TRAINING_CHARS_H__