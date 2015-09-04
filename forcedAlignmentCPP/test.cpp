#include <chrono>
#include <iostream>
#include <sstream>
#include <fstream>
#include "LearnModels.h"
#include "JonAlmazan.h"
#include "Classifier.h"
#include "Dataset.h"

using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
	Params params;

	Dataset ds(params);
	
	AnnotatedLine x;
	StartTimeSequence y;

	ds.read(x, y);
	/*
	LearnModels lm;
	lm.learnModels();
	lm.evaluateModels();
	*/

	//Classifier cls;
	//cls.loadLine("D:/Dropbox/PhD/datasets/SG/data/line_images/csg562-003-01.png");

	//JonAlmazan ja;
	//ja.LearnModelsAndEvaluate();

	cout << "Press any key to continue.";
	getchar();
	return 0;
}