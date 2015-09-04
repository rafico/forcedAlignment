#include "ForcedAlignmentTrain.h"
#include "commonTypes.h"

void static train()
{
	// phoneme symbol to number mapping (Lee and Hon, 89)
	//PhonemeSequence::load_phoneme_map(phonemes_filename, silence_symbol);

	// Initiate classifier
	//Classifier classifier(frame_rate, min_phoneme_length, max_phoneme_length, C,
	//	beta1, beta2, beta3, min_sqrt_gamma, loss_type);
	// classifier.load_phoneme_stats(phoneme_stats_filename);

	uint num_epochs = 1;

	double loss;
	double cum_loss = 0.0;
	double best_validation_loss = 1e100;
	/*
	for (uint epoch = 0; epoch < num_epochs; ++epoch) {

		// begining of the training set
		//Dataset training_dataset(scores_filelist, dists_filelist, phonemes_filelist, start_times_filelist);

		double max_loss_in_epoch = 0.0; // maximal loss value in this epoch
		double avg_loss_in_epoch = 0.0; // training loss value in this epoch

		// Run over all dataset
		for (uint i = 0; i < training_dataset.size(); i++) {

			SpeechUtterance x;
			StartTimeSequence y;
			StartTimeSequence y_hat;
			StartTimeSequence y_hat_eps;


			cout << "==================================================================================" << endl;

			// read next example for dataset
			training_dataset.read(x, y, remove_silence);
			y_hat.resize(y.size());
			y_hat_eps.resize(y.size());

			// predict label 
			classifier.predict(x, y_hat);

			cout << "phonemes=" << x.phonemes << endl;
			cout << "alignment= " << y << endl;
			cout << "predicted= " << y_hat << endl;

			if (epsilon > 0.0) {
				classifier.predict_epsilon(x, y_hat_eps, y, epsilon);
				cout << "eps-predicted= " << y_hat_eps << endl;
			}

			// suffer loss and update
			if (epsilon > 0.0) {
				loss = classifier.update_direct_loss(x, y_hat_eps, y_hat, y, epsilon);
			}
			else {
				loss = classifier.update(x, y, y_hat);
			}
			cum_loss += loss;

			if (max_loss_in_epoch < loss) max_loss_in_epoch = loss;
			avg_loss_in_epoch += loss;

			// now, check the validations error
			if (val_scores_filelist != "" && classifier.was_changed()) {
				cout << "Validation...\n";
				Dataset val_dataset(val_scores_filelist, val_dists_filelist, val_phonemes_filelist, val_start_times_filelist);
				double this_w_loss = 0.0;
				for (uint ii = 0; ii < val_dataset.size(); ++ii) {
					SpeechUtterance xx;
					StartTimeSequence yy;
					StartTimeSequence yy_hat;
					val_dataset.read(xx, yy, remove_silence, false);
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

				// stopping criterion for iterate until convergence
				//        if (best_validation_loss < 1.0)
				//          break;
			}

		} // end running over the dataset

		avg_loss_in_epoch /= training_dataset.size();

		cout << " average normalized loss = " << avg_loss_in_epoch
			<< " best validation loss  = " << best_validation_loss << endl;
	}
	if (val_scores_filelist == "")
		classifier.save(classifier_filename);
	*/
	cout << "Done." << endl;
}