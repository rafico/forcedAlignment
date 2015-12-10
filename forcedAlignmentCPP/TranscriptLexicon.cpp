#include <sstream>
#include "TranscriptLexicon.h"
#include "Dataset.h"

TranscriptLexicon::TranscriptLexicon()
	: m_params(Params::getInstance())
{
	m_training_file_list.read(m_params.m_pathSets + "train.txt");
	m_transcription_file.read(m_params.m_pathTranscription);
	sort(begin(m_transcription_file), end(m_transcription_file));
}


void TranscriptLexicon::buildLexicon()
{
	for (auto fileName : m_training_file_list)
	{
		auto p = std::lower_bound(begin(m_transcription_file), end(m_transcription_file), fileName);
		
		string docName;
		string lineId;
		string prefix;

		std::stringstream ssTranscription(*p++);
		ssTranscription >> lineId;
		docName = lineId.substr(0, lineId.find_last_of("-"));
		while (p != m_transcription_file.end() && docName.compare(fileName) == 0)
		{
			string accurateTranscriptString;
			string inaccurateTranscriptString;
			
			ssTranscription >> accurateTranscriptString >> inaccurateTranscriptString;
			
			vector<string> accurateTranscript;
			vector<string> inaccurateTranscript;

			parseAccurateTranscript(accurateTranscriptString, accurateTranscript);
			parseInAccurateTranscript(inaccurateTranscriptString, inaccurateTranscript);

			if (accurateTranscript.size() - inaccurateTranscript.size() > 1)
			{
				std::cerr << "Houston we have a problem in the transcript" << endl;
			}

			for (size_t idx = 0; idx < inaccurateTranscript.size(); ++idx)
			{
				auto &accWord = accurateTranscript[idx];
				auto &inAccWord = inaccurateTranscript[idx];

				if (0 == idx && !prefix.empty())
				{
					accWord = prefix + accWord;
					prefix.clear();
				}
				// transforming inAccWord to lower case.
				std::transform(accWord.begin(), accWord.end(), accWord.begin(), ::tolower);

				auto iter = m_lexicon.find(inAccWord);
				if (iter == end(m_lexicon))
				{
					std::set<string> firstWord{ accWord };
					m_lexicon.insert(make_pair(inAccWord, firstWord));
				}
				else
				{
					auto &wordsSet = iter->second;
					wordsSet.insert(accWord);
				}
			}

			if (accurateTranscript.size() - inaccurateTranscript.size() == 1)
			{
				prefix = accurateTranscript.back();
			}

			ssTranscription.str(*p++);
			ssTranscription.clear(); // Clear state flags.

			ssTranscription >> lineId;
			docName = lineId.substr(0, lineId.find_last_of("-"));
		}
	}
}

void TranscriptLexicon::parseAccurateTranscript(const string &transcript, vector<string> &words)
{
	string word;
	string buffer;
	for (size_t i = 0; i < transcript.size(); ++i)
	{
		char ch = transcript[i];
		switch (ch)
		{
		case '-':
		case '|':
			if (buffer == "et")
			{
				word.push_back('&');
			}
			else if (buffer == "pt")
			{
				// We ignore punctuation marks, as they are ignored in the test set.
				//word.push_back('.');
			}
			else
			{
				word.push_back(buffer[0]);
			}
			if (transcript[i] == '|')
			{
				words.push_back(word);
				word.clear();
			}
			buffer.clear();
			break;
		default:
			buffer.push_back(ch);
		}
	}
	if (buffer == "et")
	{
		word.push_back('&');
	}
	else if (buffer == "pt")
	{
		//word.push_back('.');
	}
	else
	{
		word.push_back(buffer[0]);
	}
	words.push_back(word);
}

void TranscriptLexicon::parseInAccurateTranscript(const string &transcript, vector<string> &words)
{
	string buffer;
	std::stringstream ss(transcript);
	while (std::getline(ss, buffer, '|'))
	{
		if (buffer.compare("BREAK"))
		{
			words.push_back(buffer);
		}
	}
}

void TranscriptLexicon::writeLexiconToFile()
{
	std::ofstream lexiconFile;
	lexiconFile.open("lexicon.txt");

	for (auto &wordPair : m_lexicon)
	{
		lexiconFile << wordPair.first << ": ";
		auto &inAccWords = wordPair.second;
		for (auto &inAccWord : inAccWords)
		{
			lexiconFile << inAccWord << " ";
		}
		lexiconFile << endl;
	}

	lexiconFile.close();
}

