#ifndef _H_TRANSCRIPT_LEXICON_H__
#define _H_TRANSCRIPT_LEXICON_H__

#include <set>
#include "commonTypes.h"
#include "Params.h"

class TranscriptLexicon
{
public:
	TranscriptLexicon();

	void buildLexicon();
	void writeLexiconToFile();

private:
	void parseAccurateTranscript(const string &transcript, vector<string> &words);
	void parseInAccurateTranscript(const string &transcript, vector<string> &words);
	
	std::map<string, std::set<string>> m_lexicon;
	
	StringVector m_training_file_list;
	StringVector m_transcription_file;
	const Params &m_params;
};

#endif // !_H_TRANSCRIPT_LEXICON_H__