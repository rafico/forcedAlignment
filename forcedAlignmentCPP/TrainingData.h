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
using charStatType = unordered_map<uchar, double>;

struct TrainingData
{
	TrainingData();

	void combineChars();
	const vector<Character>& getSamples(uint asciiCode);
	double getMeanWidth(uint asciiCode);
	double getMeanHeight(uint asciiCode);
	void computeNormalDistributionParams();
	void getExtermalWidths(int &maxWidth, int& minWidth);
	vector<Doc>& getTrainingDocs() { return m_trainingDocs; }
	const Doc &getDocByName(string docName);
	
	void writeQueriesAndDocsGTPfiles();

	void displayTrainingData();

	void load_char_stats(charStatType &meanCont, charStatType& stdCont) const;

	// over all characters.
	int m_globalMaxWidth;
	int m_globalMinWidth;

	TrainingCharsCont m_charInstances;
	Params& m_params;
	vector<Doc> m_trainingDocs;

	unordered_map<string, size_t> m_file2Doc;
};

#endif // !_H_TRAINING_CHARS_H__