#ifndef _H_LM_PARAMS_H__
#define _H_LM_PARAMS_H__

#include "commonTypes.h"


//TODO: replace hard-coded params with an xml or ini file.
struct LMParams
{
	LMParams()
		: m_sbin(6),
		m_rangeX(Range(-10, 12)),
		m_rangeY(Range(-10, 12)),
		m_stepSize4PositiveExamples(2),
		m_numTrWords((m_rangeX.size() / m_stepSize4PositiveExamples) * (m_rangeY.size() / m_stepSize4PositiveExamples)),
		m_propNWords(64),
		m_numNWords(m_numTrWords*m_propNWords),
		m_svmlib("jsgd"),
		m_dataset("SG"),
		m_datasetPath("/home/auser/ews/datasets/" + m_dataset + "/"),
		m_pathImages(m_datasetPath + "images/"),
		m_pathDocuments(m_datasetPath + "ground_truth/character_location/"),
		m_pathData("data/"),
		m_pathResultsParent(m_pathData + "results/"),
		m_pathResults(m_pathResultsParent + m_dataset + "/"),
		m_pathResultsImages(m_pathResults + "images/"),
		m_dim(31),
		m_step(1),
		m_thrWindows(2000),
		m_numResultImages(50),
		m_overlapnms(0.2),
		m_overlap(0.5)
	{}

	void initDirs();

	//cell size.
	uint m_sbin;
	Range m_rangeX;
	Range m_rangeY;
	uint m_stepSize4PositiveExamples;
	uint m_numTrWords;
	uint m_propNWords;
	uint m_numNWords;
	string m_svmlib;
	string m_dataset;
	string m_datasetPath;
	string m_pathImages;
	string m_pathDocuments;
	string m_pathData;
	string m_pathResultsParent;
	string m_pathResults;
	string m_pathResultsImages;
	uint m_dim;
	uint m_numDocs;
	uint m_step;
	uint m_thrWindows;
	size_t m_numResultImages;
	double m_overlapnms;
	double m_overlap;
};
#endif // !_H_LM_PARAMS_H__
