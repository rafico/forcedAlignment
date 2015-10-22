#include <iostream>
#include <fstream>
#include <iterator>
#include <sstream>
#include <boost/filesystem.hpp>
#include <boost/system/config.hpp>
#include "ForcedAlignment.h"
#include "commonTypes.h"
#include "Params.h"
#include "Classifier.h"
#include "CharClassifier.h"

using namespace boost::filesystem;

ForcedAlignment::ForcedAlignment()
	: m_classifier_filename("classifier_weight.txt")
{}

void ForcedAlignment::train()
{
	double epsilon = 0;

	string loss_type = "alignment_loss";
	string training_file_list = "training_file_list.txt";
	string validation_file_list = "validation_file_list.txt";
	string start_times_file = "charStartTime.txt";

	// Initiate classifier
	Classifier classifier(0, loss_type);

	uint num_epochs = 1;

	double loss;
	double cum_loss = 0.0;
	double best_validation_loss = 1e100;

	for (uint epoch = 0; epoch < num_epochs; ++epoch) {

		//beginning of the training set
		Dataset training_dataset(training_file_list, start_times_file);

		double max_loss_in_epoch = 0.0; // maximal loss value in this epoch
		double avg_loss_in_epoch = 0.0; // training loss value in this epoch

		// Run over all dataset
		for (uint i = 0; i < training_dataset.size(); i++)
		{
			AnnotatedLine x;
			StartTimeSequence y;
			StartTimeSequence y_hat;

			cout << "==================================================================================" << endl;

			// read next example for dataset
			training_dataset.read(x, y);
			
			y_hat.resize(y.size());

			// predict label
			classifier.predict(x, y_hat);

			cout << "alignment= " << y << endl;
			cout << "predicted= " << y_hat << endl;

			//drawSegResult(x, y, y_hat);

			loss = classifier.update(x, y, y_hat);
			cum_loss += loss;

			if (max_loss_in_epoch < loss) max_loss_in_epoch = loss;
			avg_loss_in_epoch += loss;

			// now, check the validations error
			if (classifier.was_changed()) 
			{
				cout << "Validation...\n";
				Dataset val_dataset(validation_file_list, start_times_file);
				double this_w_loss = 0.0;
				for (uint ii = 0; ii < val_dataset.size(); ++ii)
				{
					AnnotatedLine xx;
					StartTimeSequence yy;
					StartTimeSequence yy_hat;
					val_dataset.read(xx, yy);
					yy_hat.resize(yy.size());
					classifier.predict(xx, yy_hat);
					//drawSegResult(xx, yy, yy_hat);
					double this_loss = classifier.gamma(yy, yy_hat);
					this_w_loss += this_loss;
				}
				this_w_loss /= val_dataset.size();
				if (this_w_loss < best_validation_loss) 
				{
					best_validation_loss = this_w_loss;
					classifier.save(m_classifier_filename);
				}
				cout << "i = " << i << ", this validation error = " << this_w_loss
					<< ", best validation loss  = " << best_validation_loss << endl;

				// stopping criterion for iterate until convergence
				//        if (best_validation_loss < 1.0)
				//          break;
			}

		} // end running over the dataset

		avg_loss_in_epoch /= training_dataset.size();

		cout << " average normalized loss = " << avg_loss_in_epoch
			<< " best validation loss  = " << best_validation_loss << endl;
	}

	cout << "Done." << endl;
}

void ForcedAlignment::decode()
{
	const Params &params = Params::getInstance();

	string training_file_list = "test_file_list.txt";
	string loss_type = "alignment_loss";
	std::ofstream resultFile;
	std::ofstream resultFischerFile;

	path p(params.m_pathResultsImages + "alignment/");
	if (!exists(p.parent_path()))
	{
		boost::filesystem::create_directory(p.parent_path());
	}

	resultFile.open(p.string() + "result.txt", std::fstream::app);
	resultFischerFile.open(p.string() + "resultFischerFile.txt", std::fstream::out);
	
	// Initiate classifier

	Classifier classifier(0, loss_type);
	classifier.load(m_classifier_filename);
	
	// beginning of the test set
	Dataset test_dataset(training_file_list, "");

	// Run over all dataset
	for (uint i = 0; i < test_dataset.size(); i++) 
	{

		AnnotatedLine x;
		StartTimeSequence y;
		StartTimeSequence y_hat;

		cout << "==================================================================================" << endl;

		// read next example for dataset
		test_dataset.read(x, y, i);
		y_hat.resize(x.m_charSeq.size());

		// predict label 
		double confidence = classifier.predict(x, y_hat);
		if (test_dataset.labels_given())
		{
			cout << "alignment= " << y << endl;
		}
		// cout << "confidence= " << confidence << endl;
		drawSegResult(x, y_hat, params.m_pathResultsImages+"alignment/"+ x.m_lineId + ".png");
		resultFile << x.m_lineId << " " << confidence << " " << y_hat << endl;
		
		resultFischerFile << x.m_lineId << " ";
		bool startWord = true;
		for (size_t i = 1; i < y_hat.size(); ++i)
		{
			if (x.m_charSeq[i] == '|' || startWord)
			{
				char delim = startWord ? '-' : '|';
				resultFischerFile << y_hat[i] << delim;
				startWord = x.m_charSeq[i] == '|';
			}
		}
		resultFischerFile << endl;

		//cout << "aligned_phoneme_score= " << classifier.aligned_phoneme_scores(x, y_hat) << endl;
#if 0    
		int end_frame = 0;
		confidence = classifier.align_keyword(x, y_hat, end_frame);
		cout << "k_predicted= " << y_hat << " " << end_frame << endl;
		cout << "k_confidence= " << confidence << endl;
#endif    


		/*
		// calculate the error
		if (test_dataset.labels_given()) 
		{
			int file_loss = 0;
			int cur_loss;
			for (unsigned int j = 0; j < y.size(); ++j) {
				if (y[j] > y_hat[j]) {
					cur_loss = y[j] - y_hat[j];
				}
				else {
					cur_loss = y_hat[j] - y[j];
				}
				file_loss += cur_loss;
				cummulative_loss += cur_loss;
				for (int t = 1; t <= NUM_CUM_LOSS_RESOLUTIONS; t++)
					if (cur_loss <= t) cum_loss_less_than[t]++;
			}
			num_boundaries += y.size();
			cout << "File loss = " << file_loss / double(y.size()) << endl;
			cout << "Cum loss = " << cummulative_loss / double(num_boundaries) << endl;
			for (uint t = NUM_CUM_LOSS_RESOLUTIONS; t >= 1; t--) {
				cout << "% Boundaries (t <= " << t*frame_rate << "ms) = "
					<< 100.0*cum_loss_less_than[t] / double(num_boundaries) << "\n";
			}
			cout << endl;
			Classifier::loss_type = "tau_insensitive_loss";
			cummulative_loss_eps_insensitive += (Classifier::gamma(y, y_hat)*y.size());
			//      cout << "Classifier::gamma(y, y_hat)=" << Classifier::gamma(y, y_hat) << " " << num_boundaries << endl;
			cout << classifier.loss_type << "=" << cummulative_loss_eps_insensitive / double(num_boundaries) << endl;
			Classifier::loss_type = "alignment_loss";
			cummulative_loss_eps_alignment += (Classifier::gamma(y, y_hat)*y.size());
			//      cout << "Classifier::gamma(y, y_hat)=" << Classifier::gamma(y, y_hat) << " " << num_boundaries << endl;
			cout << classifier.loss_type << "=" << cummulative_loss_eps_alignment / double(num_boundaries) << endl;

		}
		*/
	}
	/*
	if (output_confidence != "" && output_confidence_ofs.good())
		output_confidence_ofs.close();
	*/
	resultFile.close();
	resultFischerFile.close();

	cout << "Done." << endl;
}

void ForcedAlignment::drawSegResult(const AnnotatedLine &x, const StartTimeSequence &y, const StartTimeSequence &y_hat)
{
	Mat img = x.m_image.clone();
	for (size_t i = 0; i < y.size(); ++i)
	{
		if (y[i] == y_hat[i])
		{
			line(img, cv::Point(y[i], 0), cv::Point(y[i], img.rows), cv::Scalar(255, 255, 0));
		}
		else
		{
			line(img, cv::Point(y[i], 0), cv::Point(y[i], img.rows), cv::Scalar(255, 0, 0));
			line(img, cv::Point(y_hat[i], 0), cv::Point(y_hat[i], img.rows), cv::Scalar(0, 255, 0));
		}
	}
}

void ForcedAlignment::drawSegResult(const AnnotatedLine &x, const StartTimeSequence &y, string resultPath)
{
	cv::Scalar green(0, 255, 0);
	cv::Scalar red(0, 0, 255);
	//cv::Scalar blue(255, 0, 0);
	Mat img = x.m_image.clone();
	bool startWord = true;
	for (size_t i = 1; i < y.size(); ++i)
	{
		if (x.m_charSeq[i] == '|' || startWord)
		{
			line(img, cv::Point(y[i], 0), cv::Point(y[i], img.rows), red);
			startWord = x.m_charSeq[i] == '|';
		}
		else
		{
			line(img, cv::Point(y[i], 0), cv::Point(y[i], img.rows), green);
		}
	}
	path p(resultPath);
	if (!exists(p.parent_path()))
	{
		boost::filesystem::create_directory(p.parent_path());
	}
	imwrite(resultPath, img);
}
