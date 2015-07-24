#include "ModelInstance.h"


/* Threshold  Ascii_code X  Y W H */
ModelInstance::ModelInstance(const std::string &fileName, const std::string &csv_line)
	: m_fileName(fileName)
{
	std::stringstream   lineStream(csv_line);
	std::string         cell;

	std::getline(lineStream, cell, ',');
	m_threshold = std::stoi(cell);

	std::getline(lineStream, cell, ',');
	m_asciiCode = std::stoi(cell);

	std::getline(lineStream, cell, ',');
	m_window.x = std::stoi(cell);

	std::getline(lineStream, cell, ',');
	m_window.y = std::stoi(cell);

	std::getline(lineStream, cell, ',');
	m_window.width = std::stoi(cell) + 1;

	std::getline(lineStream, cell, ',');
	m_window.height = std::stoi(cell) + 1;
}