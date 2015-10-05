#ifndef _H_TRAINING_DATA_H__
#define _H_TRAINING_DATA_H__

#include "commonTypes.h"
#include "Doc.h"
#include "Params.h"

struct TrainingCharsHelper
{
	double m_widthMean;
	double m_widthStd;
	double m_heightMean;
	double m_heightStd;
	int m_maxWidth;
	int m_minWidth;
	vector<Character> m_instances;
};

using TrainingCharsCont = unordered_map<uchar, TrainingCharsHelper>;

class TrainingData : public Singleton<TrainingData>
{
	friend class Singleton<TrainingData>;

private:
	TrainingData();

public:
	void combineChars();
	
	vector<uchar> getAsciiCodes();
	const vector<Character>& getSamples(uint asciiCode);
	
	void estimateNormalDistributionParams();
	double getMinWidth(uchar asciiCode);
	double getMaxWidth(uchar asciiCode);
	double getMeanWidth(uchar asciiCode);
	double getStdWidth(uchar asciiCode);
	double getMeanHeight(uchar asciiCode);
	void getExtermalWidths(vector<uchar>& charSeq, int &maxWidth, int& minWidth);
	
	vector<Doc>& getTrainingDocs() { return m_trainingDocs; }
	const Doc *getDocByName(string docName);
	
	void writeQueriesAndDocsGTPfiles();
	void displayTrainingData();

private:
	TrainingCharsCont m_charInstances;
	const Params& m_params;
	vector<Doc> m_trainingDocs;

	unordered_map<string, size_t> m_file2Doc;
};

#endif // !_H_TRAINING_CHARS_H__