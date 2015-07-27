#include "CharInstance.h"


/* Threshold  Ascii_code X  Y W H */
CharInstance::CharInstance(const string &fileName, uint globalIdx, const string &csv_line)
	: m_pathIm(fileName), m_globalIdx(globalIdx)
{
	std::stringstream   lineStream(csv_line);
	std::string         cell;

	std::getline(lineStream, cell, ',');
	m_threshold = std::stoi(cell);

	std::getline(lineStream, cell, ',');
	m_asciiCode = std::stoi(cell);

	std::getline(lineStream, cell, ',');
	m_loc.x = std::stoi(cell);

	std::getline(lineStream, cell, ',');
	m_loc.y = std::stoi(cell);

	std::getline(lineStream, cell, ',');
	m_loc.width = std::stoi(cell) + 1;

	std::getline(lineStream, cell, ',');
	m_loc.height = std::stoi(cell) + 1;
}