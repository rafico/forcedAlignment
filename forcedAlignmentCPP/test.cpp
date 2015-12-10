#include <chrono>
#include <iostream>
#include <sstream>
#include <fstream>
#include "CharClassifier.h"
#include "Classifier.h"
#include "Dataset.h"
#include "ForcedAlignment.h"
#include "Doc.h"
#include "CharSpotting.h"
#include "TranscriptLexicon.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	ForcedAlignment fa;
	fa.inAccDecode();

	//TranscriptLexicon tl;
	//tl.buildLexicon();
	//tl.writeLexiconToFile();

	//CharSpotting sp;
	//sp.evaluateModels();

	cout << "Press any key to continue.";
	getchar();
	return 0;
}