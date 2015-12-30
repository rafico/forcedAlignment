#ifndef _H_FORCED_ALIGNMENT_TRAIN_H__
#define _H_FORCED_ALIGNMENT_TRAIN_H__

#include "Dataset.h"
#include "commonTypes.h"
#include "TranscriptLexicon.h"

class ForcedAlignment
{
public:
	ForcedAlignment();
	~ForcedAlignment();

	void train();
	void decode();

	void inAccTrain();
	void inAccDecode(const TranscriptLexicon& tl);

private:
	void drawSegResult(const AnnotatedLine &x, const StartTimeSequence &y, const StartTimeSequence &y_hat);
	void drawSegResult(const AnnotatedLine &x, const StartTimeSequence &y, string resultPath);
	void printConfidencePerChar(AnnotatedLine &x, const StartTimeSequence &y);
	void writeResultInFischerFormat(std::ofstream& m_resultFischerFile, const AnnotatedLine &x, StartTimeSequence &y_hat);

	const string m_classifier_filename;
	
	const Params &m_params;

	std::map<string, string> m_resultFischerCache;
	std::ofstream m_resultFischerFile;
	std::ofstream m_resultFile;
};


#endif // !_H_FORCED_ALIGNMENT_TRAIN_H__
