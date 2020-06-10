# neurips2020_9081

This anonymised repository contains both the code and dataset used for the submission.

_Clevr-Kiwi_ generates 20 images for each scene holding a constant altitude and sampling over azimuthal angle. To ensure that the model would not have any clues as to how the view had been rotated, we replaced the asymmetrical "photo backdrop" canvas of the Clevr dataset with a large plane and centered overhead lighting. 

To focus on questions with viewpoint dependent answers, we filtered the set of questions to only include those containing spatial relationships (e.g. _is X to the right of Y?_). From the original 90 question templates, only 44 contained spatial relationships. For each spatial question template, we instantiated questions based on the canonical viewpoint of the scene. To create training and validation samples, we combined these questions with rotated and canonical images. 

In total, the training + validation split consists of 45,600 scenes, each containing roughly 10 questions for a total of 455,549 questions. 5\% of these scenes were set aside for validation. For the test set, 10,000 scenes were generated with roughly 5 questions each, for a total of 49,670 questions.

## Downloading the data

- Training + validation set: https://mega.nz/file/lCphQRzB#-9uIvp09o6IpASXYH5Ve0zy8JceJmnPBlorO3nAGgHc
- Test set: https://mega.nz/file/QOhzkJCb#4QVNbmdmcPbfyCrN-O_xQ-xnb-U_g1tR-7aLAbqOPgE

Please see the IPython notebook in `data/` for an example exploration of the dataset, including an example PyTorch Dataset class.
