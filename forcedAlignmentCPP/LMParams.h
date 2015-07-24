#ifndef _H_LMPARAMS_H__

#include "commonTypes.h"

//TODO: replace hard-coded params with an xml or ini file.
struct LMParams
{
	LMParams()
		: m_sbin(8),
		m_rangeX(Range(-10, 12)),
		m_rangeY(Range(-10, 12)),
		m_stepSize4PositiveExamples(2),
		m_propNWords(64),
		m_numTrWords((m_rangeX.size() / m_stepSize4PositiveExamples) * (m_rangeY.size() / m_stepSize4PositiveExamples)),
		m_dataset("SG"),
		m_datasetPath("D:/Dropbox/PhD/datasets/" + m_dataset+ "/"),
		m_pathQueries(m_datasetPath + "queries/"),
		m_pathImages(m_datasetPath + "images/"),
		m_pathDocuments(m_datasetPath + "documents/"),
		m_svmModelsLocation("D:/Dropbox/Code/forcedAlignmentCPP/svmModels/"),
		m_dim(31)
	{}

	//cell size.
	uint m_sbin;
	Range m_rangeX;
	Range m_rangeY;
	uint m_stepSize4PositiveExamples;
	uint m_propNWords;
	uint m_numTrWords;
	string m_dataset;
	string m_datasetPath;
	string m_pathQueries;
	string m_pathImages;
	string m_pathDocuments;
	string m_svmModelsLocation;
	uint m_dim;
};
#endif // !_H_LMPARAMS_H__
