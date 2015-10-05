#include "ForcedAlignmentTrain.h"
#include "commonTypes.h"
#include "Dataset.h"
#include "Params.h"
#include "Classifier.h"
#include "CharClassifier.h"

void drawSegResult(const AnnotatedLine &x, const StartTimeSequence &y, const StartTimeSequence &y_hat);

void ForcedAlignmentTrain::train()
{
	string val_scores_filelist;
	string classifier_filename;

	double epsilon = 0;

	string loss_type = "alignment_loss";

	// Initiate classifier
	Classifier classifier(0, loss_type);

	uint num_epochs = 1;

	double loss;
	double cum_loss = 0.0;
	double best_validation_loss = 1e100;

	for (uint epoch = 0; epoch < num_epochs; ++epoch) {

		//beginning of the training set
		Dataset training_dataset;

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

			cout << "chars=" << x.m_charSeq << endl;
			cout << "alignment= " << y << endl;
			cout << "predicted= " << y_hat << endl;

			drawSegResult(x, y, y_hat);

			loss = classifier.update(x, y, y_hat);
			cum_loss += loss;

			if (max_loss_in_epoch < loss) max_loss_in_epoch = loss;
			avg_loss_in_epoch += loss;

#ifdef USE_VALIDATION
			// now, check the validations error
			if (!val_scores_filelist.empty() && classifier.was_changed()) 
			{
				cout << "Validation...\n";
				Dataset val_dataset(val_scores_filelist, val_dists_filelist, val_phonemes_filelist, val_start_times_filelist);
				double this_w_loss = 0.0;
				for (uint ii = 0; ii < val_dataset.size(); ++ii) {
					AnnotatedLine xx;
					StartTimeSequence yy;
					StartTimeSequence yy_hat;
					val_dataset.read(xx, yy);
					yy_hat.resize(yy.size());
					classifier.predict(xx, yy_hat);
					double this_loss = 0;
					for (unsigned int jj = 0; jj < yy.size(); ++jj) {
						if (yy[jj] > yy_hat[jj]) this_loss += yy[jj] - yy_hat[jj];
						else this_loss += yy_hat[jj] - yy[jj];
					}
					this_loss /= yy.size();
					this_w_loss += this_loss;
				}
				this_w_loss /= val_dataset.size();
				if (this_w_loss < best_validation_loss) {
					best_validation_loss = this_w_loss;
					classifier.save(classifier_filename);
				}
				cout << "i = " << i << ", this validation error = " << this_w_loss
					<< ", best validation loss  = " << best_validation_loss << endl;
			}
#endif
		} // end running over the dataset

		avg_loss_in_epoch /= training_dataset.size();

		cout << " average normalized loss = " << avg_loss_in_epoch
			<< " best validation loss  = " << best_validation_loss << endl;
	}

#ifdef USE_VALIDATION
	if (val_scores_filelist.empty())
	{
		classifier.save(classifier_filename);
	}
#endif

	cout << "Done." << endl;
}

void drawSegResult(const AnnotatedLine &x, const StartTimeSequence &y, const StartTimeSequence &y_hat)
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
	int temp = 3;
}
