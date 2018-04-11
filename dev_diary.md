#### Versions and Performance

Version Score DropOut Epochs Est_Runtime_Hrs  Notes
1.0  0.67 0.35  20  5.8  First full project pipeline
1.1 0.67  0.35  20  5.8 Improved image processing pipeline for aspect ratio issues, removed flip augmentation. ~830s per epoch
1.2 0.65  0.35  20  5.6 Tried to optimize workers/batch size for keras fit_generate but wasn't able to improve really ~810s per epoch
1.3 0.57  0.35  20  2.3 Only training model on 20% of unaugmented data to increase iteration speed and increase epochs. Only ran for 20 epochs and seemed like the model might need more time to converge 
1.3a  0.59  0.35  100 11.7  Increased epochs to 100. Overkill but making sure that model has enough time to converge. Model overfits against test data around epoch 20, ~210s per epoch   
1.3b  ? 0.55  40  4.6 Increased dropout to see if that helps during the sprint mini-fitting, to then apply it to model fitted on wider data. Running for 40 epochs to give model more time to converge since dropout is so high learning is much more difficult
1.4 ? ? Improve splitter.py to v1.1 Improve Image Processing to v1.2 also (see version details)

v1.4 (planned)
_**/src/splitter.py**_ - previously splitter did StratifiedKFold on ALL of the images in the processed images directory.  So augmented images (thus far only flipped) would also end up in the validation set.  As augmentation will be ramped up majorly in this version we need to fix this undesirable behavior.
_**/src/clean_images.py**_ - Try to shoot for 10x augmentation
v1.3a
Tried going for 100 epochs, model seems to start overfitting at ~20 epochs. Runtime for 100 epochs was ~12 hours so that's not speeding anything up. However if runs only need to be ~20-25 epochs the runs would be just over 2.5 hours.
_**/src/model_cougar_v1_3.py**_ -

v1.3
Mini-training - For previous versions of the model training for 20 epochs took ~6 hours. So concept was to see how well a smaller subset of a training data performed so things could iterate faster. Unsurprisingly correct id of image class dropped, ie error increased.
_**/src/model_cougar_v1_3.py**_ - Runtime decreased to ~210s/epoch (but still ~210s for test eval so still ~420s per epoch), also 20 epochs didn't seem long enough to really evaluate

v1.2
_**/src/model_beaver_v1_2.py**_ - Tried to optimize workers and score went down slightly, probably related to removing flip augmentation

v1.1
_**/src/clean_images.py**_ - Modified to open image, consider aspect ratio, crop if the pic is rectangular, then rescale to 299x299 pixels. Also removed flip augmentation for now on 'underrepresented classes'
_**/src/model_anaconda_v1_1.py**_ - Refactored model to run as python script (instead of notebook) as this seemed to fix some notebook related weirdness w/ running tensorflow in the past.  In this case Keras .fit_generator seems to hang.  Running as python script hasn't helped, this issue still occurs sometimes.

v1.0
Created initial data ingestion, processing, and modeling pipeline:
_**/src/splitter.py**_ - extend sklearn StratifiedKfold to generate balanced class folds to evaluate model performance. 10 folds of train 90% eval 10% generated, 3 used.
_**/src/clean_images.py**_ - Open images, scale to 299x299, flip under-represented classes on vertical axis to simply augment
_**/src/loader_bot.py - Since all images won't fit in memory made a generator that gathers and feeds batches to the model during training/evaluation
_**/notebooks/first_model.ipynb**_ - Run first set of models

Cross validated at ~67% accuracy on test set after 15-20 epochs, depending on the run


3: Data Pipeline

clean_images.py Image Processing v1.0: Open images, scale to 299x299, flip under-represented classes on vertical axis to simply augment
splitter.py Stratified K-Fold Splits v1.0: extend sklearn StratifiedKfold to generate balanced class folds to evaluate model performance. 10 folds of train 90% eval 10% generated, 3 used.
loader_bot.py Data Feed Generator v1.0: Since all images won't fit in memory made a generator that gathers and feeds batches to the model during training/evaluation
Model
Using Inception V3 transfer learning
Froze Iv3 layers, dropped top layer of Iv3 and added new layers for class prediction
Inception v3 (-1000 softmax prediction final layer)
1024 FC
512 FC
256 FC
128 softmax
Currently takes ~12 hours to train
