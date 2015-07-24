#ifndef _H_LMPARAMS_H__

#include "commonTypes.h"

//TODO: replace hard-coded params with an xml or ini file.
struct LMParams
{
	LMParams()
		: m_sbin(8),
		m_rangeX(cv::Range(-10, 12)),
		m_rangeY(cv::Range(-10, 12)),
		m_stepSize4PositiveExamples(2),
		m_propNWords(64),
		m_numTrWords((m_rangeX.size() / m_stepSize4PositiveExamples) * (m_rangeY.size() / m_stepSize4PositiveExamples)),
		m_svmModelsLocation("D:/Dropbox/Code/forcedAlignmentCPP/svmModels/"),
		m_fileQueries("D:/Dropbox/Irina, Rafi and Yossi/datasets/saintgalldb-v1.0/ground_truth/character_location"),
		m_dim(31)
	{}

	//cell size.
	uint m_sbin;
	cv::Range m_rangeX;
	cv::Range m_rangeY;
	uint m_stepSize4PositiveExamples;
	uint m_propNWords;
	uint m_numTrWords;
	std::string m_svmModelsLocation;
	std::string m_fileQueries;
	uint m_dim;
};
#endif // !_H_LMPARAMS_H__
