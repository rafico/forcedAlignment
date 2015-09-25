#include <opencv2/highgui.hpp>
#include "Doc.h"
#include "HogUtils.h"
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <algorithm>

using namespace cv;
namespace pt = boost::property_tree;

/*
Given a rect [x,y,w,h],
Matlab treats I(rect) as I([x:x+w],[y:y+h]), whereas OpenCV reads it as I([x:x+w),[y:y+h)).

Therefore the following is true in Matlab for the abscissa (also for the ordinate) :
x_end = x_start+w
w = x_end - x_start

Whereas in opencv the formulas are a bit different:
x_end = x_start+w-1
w = x_end-x_start+1

To convert from w_m to w_o (matlab to opencv) use:
w_o = w_m+1

*/

DocElement::DocElement(const pt::ptree& docElem)
	: m_docNum(-1)
{
	string ID = docElem.get("ID", "");
	string ParentID = docElem.get("ParentID", "");
	string X = docElem.get("X", "");
	string Y = docElem.get("Y", "");
	string Width = docElem.get("Width", "");
	string Height = docElem.get("Height", "");

	m_ID = stoi(ID);
	m_ParentID = ParentID.empty() ? -1 : stoi(ParentID);
	m_loc.x = stoi(X) - 1;
	m_loc.y = stoi(Y) - 1;
	m_loc.width = std::stoi(Width) + 1;
	m_loc.height = std::stoi(Height) + 1;
}

uint Character::cnt = 0;

Character::Character(const pt::ptree& xmlPtree)
	: DocElement(xmlPtree), m_globalIdx(getGlobalIdx())
{
	string Transcript = xmlPtree.get("Transcript", "");
	if (Transcript.empty() || '\0' == Transcript[0])
	{
		std::cerr << "Character ID: " << m_ID << " has no transcript." << endl;
	}
	else
	{
		m_asciiCode = Transcript[0];
	}
}

Rect Character::resizeChar(const Rect &old_loc, uint sbin)
{
	// We expand the query to capture some context
	Rect loc(old_loc.x - sbin, old_loc.y - sbin, old_loc.width + 2 * sbin, old_loc.height + 2 * sbin);

	uint res;
	while ((res = (loc.height % sbin)))
	{
		loc.height -= res;
		loc.y += int(floor(double(res) / 2));
	}
	while ((res = (loc.width % sbin)))
	{
		loc.width -= res;
		loc.x += int(floor(double(res) / 2));
	}

	return loc;
}

Doc::Doc()
{}

Doc::Doc(string pathImage)
{
	Init(pathImage);
}

void Doc::Init(string pathImage)
{
	m_origImage = imread(pathImage);
	m_pathImage = pathImage;
	if (!m_origImage.data)
	{
		std::cerr << "Could not open or find the image " << m_pathImage << std::endl;
	}
	m_H = m_origImage.rows;
	m_W = m_origImage.cols;
}

void Doc::resizeDoc(uint sbin)
{
	uint H = m_H;
	uint W = m_W;
	uint res;

	while ((res = (H % sbin)))
	{
		H -= res;
	}
	while ((res = (W % sbin)))
	{
		W -= res;
	}
	uint difH = m_H - H;
	uint difW = m_W - W;
	uint padYini = difH / 2;
	uint padYend = difH - padYini;
	uint padXini = difW / 2;
	uint padXend = difW - padXini;

	const Mat &im = m_origImage;
	im(Range(padYini, im.rows - padYend), Range(padXini, im.cols - padXend)).copyTo(m_image);
	m_yIni = padYini;
	m_xIni = padXini;
	m_H = m_image.rows;
	m_W = m_image.cols;
}

void Doc::computeFeatures(uint sbin)
{
	resizeDoc(sbin);
	Mat feat = HogUtils::process(m_image, sbin, &m_bH, &m_bW);
	feat.convertTo(m_features, CV_32F);
}

/*
<DocumentElement>
<ID>99673</ID>
<ParentID>208719</ParentID>
<ElementType>Character</ElementType>
<X>470</X>
<Y>2062</Y>
<Width>32</Width>
<Height>40</Height>
<Transcript>e</Transcript>
<Threshold>174</Threshold>
<OriginX>488</OriginX>
<OriginY>2100</OriginY>
</DocumentElement>
*/

// For debugging.
void display_ptree(pt::ptree const& pt)
{
	for(const auto& v : pt)
	{
		std::cout << v.first << ": " << v.second.get_value<std::string>() << "\n";
		display_ptree(v.second);
	}
}

void Doc::loadXml(const string& pathXml)
{
	pt::ptree tree;

	// Parse the XML into the property tree.
	try
	{
		pt::read_xml(pathXml, tree);
	}
	catch (pt::xml_parser_error& e)
	{
		std::cerr << e.message() << ": " << e.filename() << endl;
		throw;
	}

	unordered_map<size_t, int> char2WordsIndices;
	unordered_map<size_t, int> word2LineIndices;
	unordered_map<int, size_t> Id2Indices;

	for(auto &v : tree.get_child("ArrayOfDocumentElement"))
	{
		auto& docElem = v.second;
		//display_ptree(docElem);

		string ID = docElem.get("ID", "");
		if (!ID.empty())
		{
			string ElementType = docElem.get("ElementType", "");

			if ("Character" == ElementType)
			{
				Character ch(docElem);
				if (!ch.m_ParentID)
				{
					std::cerr << "Character ID: " << ch.m_ID << " has no parent !" << endl;
				}
				else
				{
					char2WordsIndices.insert({ m_chars.size(), ch.m_ParentID });
					m_chars.push_back(ch);
				}
			}
			else if ("Word" == ElementType)
			{
				Word w(docElem);
				if (!w.m_ParentID)
				{
					std::cerr << "Word ID: " << w.m_ID << " has no parent !" << endl;
				}
				else
				{
					Id2Indices.insert({ w.m_ID, m_words.size()});
					word2LineIndices.insert({ m_words.size(), w.m_ParentID });
					m_words.push_back(w);
				}
			}
			else if ("Line" == ElementType)
			{
				Line l(docElem);
				Id2Indices.insert({ l.m_ID, m_lines.size()});
				m_lines.push_back(l);
			}
		}
	}

	for (size_t chIdx = 0; chIdx < char2WordsIndices.size(); ++chIdx)
	{
		auto wordId = Id2Indices[char2WordsIndices[chIdx]];
		Word& word = m_words[wordId];
		word.m_charIndices.push_back(chIdx);
	}

	for (size_t wordIdx = 0; wordIdx < word2LineIndices.size(); ++wordIdx)
	{
		auto lineID = Id2Indices[word2LineIndices[wordIdx]];
		Line& line = m_lines[lineID];
		line.m_wordIndices.push_back(wordIdx);
	}
	sortElements();
}

void Doc::sortElements()
{
	// sort according to location
	std::sort(m_lines.begin(), m_lines.end(), [](const Line& lhs, const Line& rhs){return lhs.m_loc.y < rhs.m_loc.y; });

	for (auto& line : m_lines)
	{
		std::sort(line.m_wordIndices.begin(), line.m_wordIndices.end(), [&](size_t lhs, size_t rhs)
		{
			return m_words[lhs].m_loc.x < m_words[rhs].m_loc.x;
		});

		for (auto& wordIdx : line.m_wordIndices)
		{
			Word &word = m_words[wordIdx];
			std::sort(word.m_charIndices.begin(), word.m_charIndices.end(), [&](size_t lhs, size_t rhs)
			{
				return m_chars[lhs].m_loc.x < m_chars[rhs].m_loc.x;
			}
			);
		}
	}
}

std::ostream& operator<< (std::ostream& os, const Doc& doc)
{
	for (size_t lineId = 0; lineId < doc.m_lines.size(); ++lineId)
	{
		os << lineId << ": ";
		auto& line = doc.m_lines[lineId];

		for (auto wordId :line.m_wordIndices)
		{
			auto& word = doc.m_words[wordId];
			for (auto ch : word.m_charIndices)
			{
				auto charIns = doc.m_chars[ch];
				os << charIns.m_asciiCode;
			}
			os << " ";
		}
		os << endl;
	}
	return os;
}