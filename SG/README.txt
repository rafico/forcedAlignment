======= TERMS OF USAGE ======= 

The Saint Gall database may be used for non-commercial research and teaching purposes only. If you are publishing scientific work based on the Saint Gall database, we request you to include a reference to our paper:

A. Fischer, V. Frinken, A. Fornés, and H. Bunke: "Transcription Alignment of Latin Manuscripts using Hidden Markov Models," in Proc. 1st Int. Workshop on Historical Document Imaging and Processing, pages 29-36, 2011.

======= DATA SET =======

* IDs

The text line ID "csg562-003-01" can be read as follows: manuscript ID (Codex Sangallensis 562), page 3, line 1. The page ID is "csg562-003" accordingly. Line numbers start at 1.

* sets/

Disjoint sets for training, validation, and testing given by page IDs. These sets were used in our publication.

* data/page_images

300dpi JPG images of the original manuscript pages.

* data/line_images_normalized

PNG images of binarized and normalized (skew and height) text lines.

* data/edition.txt

[pageID] [label_1]|...|[label_n]

Word labels of the original text edition (J.-P. Migne PL114) for each page. Note that word spelling, capitalization, and punctuation deviate from the manuscript images.

* ground_truth/line_location

SVG paths of the text line locations for each page. For each text line, a single closed path is defined.

* ground_truth/word_location.txt

[lineID] [line width] [start_1-end_1]|...|[start_n-end_n]

Start and end positions of the words for each normalized text line image in the validation and test set. The positions are rather tight around the words, i.e., they may be shifted about half a character width towards the word center.

* ground_truth/transcription.txt

[lineID] [spelling_1]|...|[spelling_n] [label_1]|...|[label_n]

Text line transcription with accurate spellings that correspond with the manuscript images. Spellings are given by:

[character_1]-...-[character_m]

Besides lower case and upper case letters, we have also employed two special letters, i.e., "pt" for punctuation marks (typically dots between words) and "et" for the special character "&" that is frequently used throughout the data set. The word spellings are followed by the corresponding labels from the text edition. The labels are lower case only and without punctuation marks.

======= CONTACT INFORMATION ======= 

If you have any questions or suggestions, please contact Andreas Fischer (afischer@iam.unibe.ch).
