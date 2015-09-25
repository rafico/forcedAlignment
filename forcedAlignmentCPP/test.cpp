#include <chrono>
#include <iostream>
#include <sstream>
#include <fstream>
#include "CharClassifier.h"
#include "Classifier.h"
#include "Dataset.h"
#include "ForcedAlignmentTrain.h"
#include "Doc.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	
	//Doc doc;
	//doc.loadXml("D:/Dropbox/PhD/datasets/SG/ground_truth/csg562-005.xml");

	//cout << doc;

	//ForcedAlignmentTrain::train();

	CharClassifier cc;
	//cc.learnModels();
	cc.evaluateModels(false);
	int x = 2;

	cout << "Press any key to continue.";
	getchar();
	return 0;
}