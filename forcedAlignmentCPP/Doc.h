#ifndef _H_DOC_H__
#define _H_DOC_H__

#include "commonTypes.h"
#include <boost/property_tree/ptree.hpp>

namespace pt = boost::property_tree;

// class for holding both document images and text line images.

struct DocElement
{
	DocElement(const pt::ptree& docElem);
	
	Rect m_loc;
	size_t m_docNum;

	// inner variables for parsing the xml.
	int m_ID;
	int m_ParentID;
};

struct Character : DocElement
{
	Character(const pt::ptree& xmlPtree);
	
	static Rect resizeChar(const Rect &old_loc, uint sbin);
	
	uint m_globalIdx;
	uchar m_asciiCode;

private:
	static uint getGlobalIdx() { return cnt++; }
	static uint cnt;
};

struct Word : DocElement
{
	Word(const pt::ptree& xmlPtree)
		: DocElement(xmlPtree)
	{}
	vector<size_t> m_charIndices;
};

struct Line : DocElement
{
	Line(const pt::ptree& xmlPtree)
		: DocElement(xmlPtree)
	{}
	vector<size_t> m_wordIndices;
};

struct Doc
{
	Doc();
	Doc(string pathImage);
	void Init(string pathImage);

	void loadXml(const string& pathXml);
	void sortElements();

	void resizeDoc(uint sbin);
	void computeFeatures(uint sbin);
	void getComputedFeatures(Mat &features, int &BH, int&BW, uint sbin);

	string m_pathImage;
	Mat m_origImage;
	Mat m_image;
	Mat m_features;	
	uint m_H;	
	uint m_W;
	uint m_yIni;
	uint m_xIni;
	int m_bH;
	int m_bW;

	bool m_featuresComputed;

	vector<Character> m_chars;
	vector<Word> m_words;
	vector<Line> m_lines;
};

std::ostream& operator<< (std::ostream& os, const Doc& doc);

#endif // !_H_DOC_H__
