#include "CharInstance.h"


// Threshold  Ascii_code X Y W H
// This is Matlab based indexing (starts from 1).

CharInstance::CharInstance(const string &pathIm, size_t docNum, uint globalIdx, const string &csv_line)
	: m_pathIm(pathIm), m_docNum(docNum), m_globalIdx(globalIdx)
{
	std::stringstream   lineStream(csv_line);
	std::string         cell;

	std::getline(lineStream, cell, ',');
	m_threshold = std::stoi(cell);

	std::getline(lineStream, cell, ',');
	m_asciiCode = std::stoi(cell);

	std::getline(lineStream, cell, ',');
	m_loc.x = std::stoi(cell)-1;

	std::getline(lineStream, cell, ',');
	m_loc.y = std::stoi(cell)-1;

	std::getline(lineStream, cell, ',');
	m_loc.width = std::stoi(cell) + 1;

	std::getline(lineStream, cell, ',');
	m_loc.height = std::stoi(cell) + 1;
}