#include <sstream>
#include "commonTypes.h"
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
	std::map<string, std::set<string>> m_lexiconTemp;

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

				auto iter = m_lexiconTemp.find(inAccWord);
				if (iter == end(m_lexiconTemp))
				{
					std::set<string> firstWord{ accWord };
					m_lexiconTemp.insert(make_pair(inAccWord, firstWord));
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
	
	for (auto &iter : m_lexiconTemp)
	{
		vector<string> tempVec(iter.second.begin(), iter.second.end());
		m_lexicon.insert({ iter.first, tempVec });
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

std::vector<string> TranscriptLexicon::getSynonymous(const string &word) const
{
	auto iter = m_lexicon.find(word);
	if (iter == m_lexicon.end())
	{
		return { word };
	}
	else
	{
		return iter->second;
	}
}

std::ostream& operator<< (std::ostream& os, const vector<int>& y)
{
	for (uint i = 0; i < y.size(); i++)
		os << y[i] << " ";

	return os;
}

vector<CharSequence> TranscriptLexicon::getPossibleLineVariations(const CharSequence& char_seq) const
{
	string word;
	vector<string> words;
	for (size_t i = 1; i < char_seq.size(); ++i)
	{
		if (char_seq[i] != '|')
		{
			word.push_back(char_seq[i]);
		}
		else
		{
			words.push_back(word);
			word.clear();
		}
	}

	int possibilities = 1;
	vector<int> PossibilitiesVec(words.size(), 0);

	for (size_t idx = 0; idx < words.size(); ++idx)
	{
		auto &word = words[idx];
		auto Synonymous = getSynonymous(word);
		size_t sz = Synonymous.size();
		PossibilitiesVec[idx] = sz;
		possibilities *= sz;
		if (sz != 1)
		{
			std::cout << word << ": ";
			for (auto &s : Synonymous)
			{
				std::cout << s << ", ";
			}
		}
	}

	std::cout << "\nNumer of variations is :" << possibilities << endl;
	vector<CharSequence> result;

	vector<int> enumerationVec(words.size(), 1);
	for (int enumeration = 0; enumeration < possibilities; ++enumeration)
	{
		CharSequence charSeq;
		charSeq.push_back('|');

		bool carry = true;
		for (size_t idx = 0; idx < words.size(); ++idx)
		{
			auto &word = words[idx];
			auto Synonymous = getSynonymous(word);
			string syn = Synonymous[enumerationVec[idx]-1];
			charSeq.insert(end(charSeq), begin(syn), end(syn));
			charSeq.push_back('|');

			if (carry && (enumerationVec[idx] == PossibilitiesVec[idx]))
			{
				enumerationVec[idx] = 1;
			}
			else if (carry)
			{
				enumerationVec[idx]++;
				carry = false;
			}
			}
		result.push_back(charSeq);
		}


	return result;
}
