#ifndef _H_PARAMS_H__
#define _H_PARAMS_H__

#include "commonTypes.h"
#include "Singleton.h"

//TODO: replace hard-coded params with an xml or ini file.
class Params : public Singleton<Params>
{
	friend class Singleton<Params>;

private:
	Params()
		: m_sbin(6),
		m_rangeX(Range(-4, 6)),
		m_rangeY(Range(-4, 6)),
		m_stepSize4PositiveExamples(2),
		m_numTrWords((m_rangeX.size() / m_stepSize4PositiveExamples) * (m_rangeY.size() / m_stepSize4PositiveExamples)),
		m_propNWords(16),
		m_numNWords(m_numTrWords*m_propNWords),
		m_svmlib("liblinear"),
		m_dataset("SG"),
		m_dim(31),
		m_step(1),
		m_thrWindows(2000),
		m_numResultImages(50),
		m_overlapnms(0.2),
		m_overlap(0.5),

#if _WIN32 || _WIN64
		m_datasetPath("D:/datasets/" + m_dataset + "/"),
#elif __GNUC__
		m_datasetPath("/home/auser/datasets/" + m_dataset + "/"),
#endif
		m_pathImages(m_datasetPath + "images/"),
		m_pathGT(m_datasetPath + "ground_truth/"),
		m_pathCharLocation(m_pathGT + "char_location/"),
		m_pathData("data/"),
		m_pathResultsParent(m_pathData + "results/"),
		m_pathResults(m_pathResultsParent + m_dataset + "/"),
		m_pathResultsImages(m_pathResults + "images/"),
		m_pathCharModels(m_pathData + "charModels" + "_" + m_dataset + "_" + "sbin" + std::to_string(m_sbin) + "/"),
		m_pathTranscription(m_pathGT + "transcription.txt"),
		m_pathStartTime(m_pathGT + "charStartTime.txt"),
		m_pathLineImages(m_datasetPath + m_pathData + "line_images/"),
		m_pathLineBinImages(m_datasetPath + m_pathData + "line_images_binarized/"),
		m_pathSets(m_datasetPath + "sets/")
	{
		initDirs();
	}

	void initDirs();

public:
	
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
	uint m_dim;
	uint m_step;
	uint m_thrWindows;
	size_t m_numResultImages;
	double m_overlapnms;
	double m_overlap;

	string m_datasetPath;
	string m_pathImages;
	string m_pathGT;
	string m_pathCharLocation;
	string m_pathData;
	string m_pathResultsParent;
	string m_pathResults;
	string m_pathResultsImages;
	string m_pathCharModels;
	string m_pathTranscription;
	string m_pathStartTime;
	string m_pathLineImages;
	string m_pathLineBinImages;
	string m_pathSets;
};
#endif // !_H_PARAMS_H__
