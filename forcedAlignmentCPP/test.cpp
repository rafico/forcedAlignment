#include <chrono>
#include <iostream>
#include <sstream>
#include <fstream>
#include "CharClassifier.h"
#include "Classifier.h"
#include "Dataset.h"
#include "ForcedAlignmentTrain.h"
#include "Doc.h"
#include "CharSpotting.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	
	ForcedAlignmentTrain::train();

	//CharSpotting sp;
	//sp.evaluateModels();

	cout << "Press any key to continue.";
	getchar();
	return 0;
}