#### Versions and Performance

`V:` version
`Acc:` accuracy
`DO:` dropout
`LR:` learning rate
`E:` epochs
`IPE:` images per epoch
`HPR:` hours per run

| V | Acc | DO | LR | E | IPE | HPR | Notes |
|:---:|:---:|:---:|:----:|:---:|:----:|:----:|:-------------------------------------------------------------------|
| 1.0 | 0.67 | 0.35 | 0.0135 | 50 | 180k | 14.4 | First full project pipeline |
| 1.1 | 0.67 | 0.35 | 0.0135 | 20 | 162k | 5.8 | Improved image processing pipeline for aspect ratio issues, removed flip augmentation. ~830s per epoch |
| 1.2 | 0.65 | 0.35 | 0.0135 | 20 | 162k | 5.6 | Tried to optimize workers/batch size for keras fit_generate but wasn't able to improve really ~810s per epoch |
| 1.3 | 0.57 | 0.35 | 0.0135 | 20 | 18k | 1.9 | Only training model on 20% of unaugmented data to increase iteration speed and increase epochs. Only ran for 20 epochs and seemed like the model might need more time to converge |
| 1.3a | 0.59 | 0.35 | 0.0135 | 100 | 18k | 9.3 | Increased epochs to 100. Overkill but making sure that model has enough time to converge. Model overfits against test data around epoch 20, ~210s per epoch |
| 1.3b | 0.56 | 0.55 | 0.0135 | 40 | 18k | 3.7 | Increased dropout to see if that helps during the sprint mini-fitting, to then apply it to model fitted on wider data. Running for 40 epochs to give model more time to converge since dropout is so high learning is much more difficult |
| 1.3c | 0.58 | 0.55 | 0.0135 | 60 | 18k | 5.6 | Increased epochs of 1.3b to 60 as it looked like test loss was plateauing but want to be sure. Changed weight initialization back to he_normal to see if the model converges faster since epochs 1 - 40 should be the same ground |
| 1.3d | 0.59 | 0.55 | 0.0135 | 100 | 18k | 9.3 | Increased epochs to 100. Overkill but making sure that model has enough time to converge with higher dropout. Back to 'he_normal' weight initialization to see if it helps. |
| 1.4 | 0.71 | 0.45 | 0.0135 | 30 | 360k | 13.3 | Image augmentation flip, dropout down to 0.45, score prob higher than it should be since flipped images are still in the test set |
| 1.5a | .53 | 0.55 | 0.001 | 40 | 18k | 3.7 | Experiment with Adam optimizer |
| 1.5b | .15 | 0.55 | 0.005 | 40 | 18k | 3.7 | Adam optimizer increased learning rate and model never converged |
| 1.5c | 0.57 | 0.55 | 0.0005 | 40 | 18k | 3.7 | Adam optimizer decreased learning rate and...  |
| 1.6 | ? | 0.55 | 0.00025 | 40 | 18k | 3.7 | Two stage, use frozen Iv3 model for 20 epochs then unfreeze top 7 layers (172 frozen to 165 frozen)  |


#### v1.7
* _**/src/clean_images.py**_ - Try to shoot for 10x augmentation

#### v1.6
* _**`/src/model_echidna_v1_5a.py`**_ - Fine tune model after some break in. After 20 epochs go from 172 (all) frozen Iv3 layers to 165 frozen layers.

#### v1.5a/b
* _**`/src/model_echidna_v1_5a.py`**_ - Take model v1_3b for rapid prototyping. Experiment with tuning the optimizer to see if we can get Adam working (better) than SGD that has been used so far.

#### v1.4b
* _**/src/splitter.py**_ - previously splitter did StratifiedKFold on ALL of the images in the processed images directory.  So augmented images (thus far only flipped) would also end up in the validation set.  As augmentation will be ramped up majorly in this version we need to fix this undesirable behavior.
* Splitter now creates a dictionary that is serialized to a json file.  Test sets contain no augmented images, train set contains all augmented and normal images, but no test set images (augmented or otherwise)
* Verified that models are training properly after loading train/test set indices from JSON file
* Attempted to change optimizer from SGD to Adam but reverted after model diverged
* Models not run to completion so no data logged as far as learning rate, accuracy, etc

#### v1.4
<img src="/imgs/model_v1_4.png" alt="Model v1.4" width="800" height="400">

* Went back to normal 'full data' of split. Best score yet. Augmented 2x by flip over vertical axis. Score is probably a bit artificially high because the splitting code hasn't yet been updated to withhold augmented images.
* _**/src/clean_images.py**_ - Simply modified to flip augment all pictures, but not handling the issue of say the model training on normal version of an image and validating on the flipped version.  Will update again for v1.5

#### Comparison of default (Xavier) model weight initialization vs 'he_normal'

<img src="/imgs/model_xavier_v_he_normal.png" alt="Model v1.3d" width="800" height="400">

* Model v1.3c and v1.3d the only difference is epoch length and then the type of initialization.  Using PS to overlay the Xavier over the He_normal (Xavier in red) it seems like Xavier overfits slightly less and has slightly better accuracy, in this case.  Not really enough to worry about now but using default seems better.

#### v1.3d
<img src="/imgs/model_v1_3d.png" alt="Model v1.3d" width="800" height="400">

* Increased epochs from 60 to 100 to give more time to converge/verify that it's over fitting.  Also changed weight initialization from default to 'he_normal'

#### v1.3c
<img src="/imgs/model_v1_3c.png" alt="Model v1.3b" width="800" height="400">

* Increased dropout to relatively absurd level of 0.55. (But had success with that value in the past) Increased epochs to 40 because convergence should be slower since it's much more difficult to learn. This is confirmed in the above chart but it also seems like a few more epochs may show if the accuracy can improve.

#### v1.3b
<img src="/imgs/model_v1_3b.png" alt="Model v1.3b" width="800" height="400">

* Increased dropout to relatively absurd level of 0.55. (But had success with that value in the past) Increased epochs to 40 because convergence should be slower since it's much more difficult to learn. This is confirmed in the above chart but it also seems like a few more epochs may show if the accuracy can improve.

#### v1.3a
<img src="/imgs/model_v1_3a.png" alt="Model v1.3a" width="800" height="400">

* Tried going for 100 epochs, model seems to start overfitting at ~20 epochs. Runtime for 100 epochs was ~12 hours so that's not speeding anything up. However if runs only need to be ~20-25 epochs the runs would be just over 2.5 hours.

#### v1.3
<img src="/imgs/model_v1_3.png" alt="Model v1.3" width="800" height="400">

* Mini-training - For previous versions of the model training for 20 epochs took ~6 hours. So concept was to see how well a smaller subset of a training data performed so things could iterate faster. Unsurprisingly correct id of image class dropped, ie error increased.
* _**/src/model_cougar_v1_3.py**_ - Runtime decreased to ~210s/epoch (but still ~210s for test eval so still ~420s per epoch), also 20 epochs didn't seem long enough to really evaluate

#### v1.2
<img src="/imgs/model_v1_2.png" alt="Model v1.2" width="800" height="400">

* _**/src/model_beaver_v1_2.py**_ - Tried to optimize workers and score went down slightly, probably related to removing flip augmentation

#### v1.1
<img src="/imgs/model_v1_1.png" alt="Model v1.1" width="800" height="400">

* _**/src/clean_images.py**_ - Modified to open image, consider aspect ratio, crop if the pic is rectangular, then rescale to 299x299 pixels. Also removed flip augmentation for now on 'underrepresented classes'
* _**/src/model_anaconda_v1_1.py**_ - Refactored model to run as python script (instead of notebook) as this seemed to fix some notebook related weirdness w/ running tensorflow in the past.  In this case Keras .fit_generator seems to hang.  Running as python script hasn't helped, this issue still occurs sometimes.

#### v1.0
<img src="/imgs/brian_model_lr_0_005_ep_50_batch_256_dropout_0_15.png" alt="First Model" width="800" height="400">

* Created initial data ingestion, processing, and modeling pipeline:
* _**/src/splitter.py**_ - extend sklearn StratifiedKfold to generate balanced class folds to evaluate model performance. 10 folds of train 90% eval 10% generated, 3 used.
* _**/src/clean_images.py**_ - Open images, scale to 299x299, flip under-represented classes on vertical axis to simply augment
* _**/src/loader_bot.py**_ - Since all images won't fit in memory made a generator that gathers and feeds batches to the model during training/evaluation
* _**/notebooks/first_model.ipynb**_ - Run first set of models
* Cross validated at ~67% accuracy on test set after 15-20 epochs, depending on the run
