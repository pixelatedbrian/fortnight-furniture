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
| 1.5a | 0.53 | 0.55 | 0.001 | 40 | 18k | 3.7 | Experiment with Adam optimizer, needs some tuning |
| 1.5b | 0.15 | 0.55 | 0.005 | 40 | 18k | 3.7 | Adam optimizer increased learning rate and model never converged |
| 1.5c | 0.57 | 0.55 | 0.0005 | 40 | 18k | 3.7 | Adam optimizer decreased learning rate and... did the best of these models  |
| 1.6 | 0.72 | 0.55 | 0.00025 | 40 | 18k | 3.7 | Two stage, use frozen Iv3 model for 20 epochs then unfreeze top 7 layers (172 frozen to 165 frozen) New record accuracy 0.72  |
| 1.6b | 0.74 | 0.55 | 0.00025 | 40 | 18k | 3.7 | Four stage, use frozen Iv3 model for 5 epochs then unfreeze top 2 layers for each of remaining 3 rounds. Also decrease learning rate by LR/2^round New record accuracy 0.72  |
| 1.7a | 0.80 | 0.55 | 0.0001 | 20 | 360k | 9.3 | Defrosting on fuller data pipeline, all non-split pics with 2x augmentation from flip over vertical axis. New record 0.80 accuracy. |
| 1.7b | 0.79 | 0.55 | 0.0001 | 25 | 360k | 9.3 | Made initial training of frozen model + addon for double epochs (10) then three stages of 5 epochs with more layers unfreezing per mini-train |
| 1.7c | 0.80 | 0.55 | 0.00025 | 24 | 360k | 16.1 | Dropped mini-trains to 3 batches, made first train the same number of epochs as other mini-trains. Raised LR some and kept it stable. |
| 1.7d | 0.80 | 0.55 | 0.00025 | 12 | 360k | 6.0 | Dropped mini-trains to 2 batches, only went 6 epochs per mini-train. Unthawing a ~6 layers at once didn't seem to help. |
| 1.7e | 0.82 | 0.55 | 0.00025 | 12 | 360k | 5.6 | Increased mini-batches to 4, only went 3 epochs per mini-train. Unthawing 2 layers per mini-train after the initial pretrain. Also went back to dividing LR by 2^(mini-train - 1) so mini-train 3 will be LR / 4.0 New record accuracy. |
| 1.8a | 0.81 | 0.55 | 0.0000625 | 24 | 360k | 11.6 | Same as 1.7e but 6 epochs per mini-train. Drop starting LR to 0.0000625 |
| 1.8b | 0.817 | 0.55 | 0.00025 | 12 | 360k | 5.7 | Same as 1.7e but increase the amount of layers thawed per mini-train from 2 to 3 |
| 2.0a | 0.603 | 0.25 | 0.00075 | 20 | 18k | 1.5 | model 1_6 core with sprint training but using augmentation of zooming and rotation. Proof of concept, finally debugged |
| 2.0b | 0.738 | 0.25 | 0.00025 | 40 | 18k | 5.6 | model 2_0a with unfrozen layers like 1.6b and revised learning rate |
| 2.0c | 0.73 | 0.55 | 0.00025 | 80 | 18k | 11.1 | model 2_0b with more dropout and double epochs |
| 2.1a | 0.74 | 0.55 | 0.0005 | 80 | 18k | 5.6 | went to lighter NN, only 1024 FC layer |
| 2.1b | ? | 0.55 | 0.0005 | 80 | 18k | ? | model 2_1b with 512 FC layer only |

#### v2.0d (sprint)
<img src="/imgs/model_v2_0d.png" alt="Model v2_0d" width="800" height="400">
  * Model v2.0b but with increased dropout (from 0.25 to 0.55) and increased learning rate

#### v2.0c (sprint)
<img src="/imgs/model_v2_0c.png" alt="Model v2_0c" width="800" height="400">
  * Model v2.0b but with increased dropout (from 0.25 to 0.55) and double total epochs (from 40 to 80 total)

#### v2.0b (sprint)
<img src="/imgs/model_v2_0b.png" alt="Model v2_0b" width="800" height="400">
  * Model v2.0a but with layer unfreezing
  * Relatively low dropout of 0.25 still
  
#### v2.0a (sprint)
<img src="/imgs/model_v2_0a.png" alt="Model v2_0a" width="800" height="400">
  * Figured out how to augment with loader_bot.py so going to make a new version of loader_bot that will augment on the fly.
  * Will need to define how much augmentation per epoch to use, probably an optional init variable to pass in
  * Then use the quick batch to train a model on a relatively small amount of images but with about 10x augmentation
  * Also, the augmentation developed uses the larger edge to randomly select a zoomed subset of the image, potentially with some rotation. (ie if, as is typical, the width is bigger than the height) then the position of the square from the image will come from the width, this helps give the crop access to the sides of the image a bit better. (Although unclear if that will actually help.)
  * Found that some training images are 1px by 1px, which is a big problem.  Determining just how many files are affected like this. May need to attempt to scrape these poison files in a more intelligent manner.

# First official contest submissions from model trained off of model v1.8b
  * Added FireBot class to loader_bot.py which loads items to be predicted, but not trained, in order to get predictions on test images.
  * Fixed a bug in FireBot/LoaderBot in which the last batch was not run properly so not all items were predicted. (Need to report the bug/fix to the Stanford person who posted the original inspiration code.) 
  * _**/src/oracle.py**_ - Made a script that loads model weights, loads the test images, and then gets predictions on those images.
  * _**/notebooks/prediction_merging_for_submission.ipynb**_ - a note book to take the predictions from oracle.py and convert them to submitable form.
  * Scored 0.79 accuracy on leaderboard and rank ~75/220

#### v1.8b
<img src="/imgs/model_v1_8b.png" alt="Model v1.8b" width="800" height="400">

* _**`/src/model_hippogriff_v1_8.py`**_ - Modify model v1.8b slightly from model v1.7e:
  * model v1.7e thawed 2 layers per mini-train. Increasing this to 3 per mini-train. Otherwise should be the same as model 1.7e

#### v1.8a
<img src="/imgs/model_v1_8a.png" alt="Model v1.8a" width="800" height="400">

* _**`/src/model_hippogriff_v1_8.py`**_ - Modify model v1.8a slightly from model v1.7e:
  * Very close to the 0.82 accuracy number.
  * Taking 1.7 model and trying to slow it down some, double epochs, set starting LR to 1/4th of model 1_7e
  * Model may not converge, but I suspect that once the later layers start to open up that it will be ok.
  * Got to 0.812 accuracy. Slowing down didn't seem to help, but oc took twice as long.

#### v1.7e
<img src="/imgs/model_v1_7e.png" alt="Model v1.7e" width="800" height="400">

* _**`/src/model_goat_v1_7.py`**_ - Modify model v1.7e slightly from model v1.7d:
  * Struggling to see the 0.82 accuracy number again.
  * Going back to 1.6b/1.7a methodology as the more recent experiments haven't proven out.
  * New record accuracy of 0.816

#### v1.7d
<img src="/imgs/model_v1_7d.png" alt="Model v1.7d" width="800" height="400">

* _**`/src/model_goat_v1_7.py`**_ - Modify model v1.7d slightly from model v1.7c:
  * Seems to be no need to train first stage more than 6-7 epochs because the curve seems to stabilize right there.
  * Thawing small amounts of layers (1-2 layers) per mini-train doesn't seem to be helping. Basically there is a big jump from the first unfreeze and then after that it's just helping the model overfit.
  * Therefore this run trying to unfreeze a bunch of layers but only for one mini-epoch.  As an additional benefit it should run faster because not as many total epochs. Ran into the generating hang again on 1.7c so shorter runs will help reduce the probability of that occurring.
  * Therefore going from 3 x 8 epochs to 2 x 6 for this run.

#### v1.7c
<img src="/imgs/model_v1_7c.png" alt="Model v1.7c" width="800" height="400">

* _**`/src/model_goat_v1_7.py`**_ - Modify model v1.7c slightly from model v1.7b:
  * Raise LR to 0.00025 because that was the LR of the 'accidental' but ephemereal 0.82 acc
  * Kept learning rate constant between mini-trains
  * Doubling the first stage train epochs didn't seem to help, as one can see looking at these charts, so went back to same number of epochs for all minibatches but increased epochs per mini-train from 5 epochs to 8.

#### v1.7b
<img src="/imgs/model_v1_7b.png" alt="Model v1.7b" width="800" height="400">

* _**`/src/model_goat_v1_7.py`**_ - Modify model v1.7b slightly from model v1.7:
  * First increase lead in pre-training from 5 epochs to 10 epochs.
  * Also decrease initial learning rate of unfrozen stages but also keep it constant. Trying reducing LR by an order of magnitude (LR / 10) because it seems like as layers thaw the model wants to overfit very quickly.

#### v1.7a
<img src="/imgs/model_v1_7.png" alt="Model v1.7a" width="800" height="400">

* _**`/src/model_goat_v1_7.py`**_ - Take 1.6b architecture with a planned 20 epoch run of 4 mini-trains of 5 epochs each. Use the same decaying LR and layer thawing. This time, however, train on ~360k images, normal and flip augmented.  Run time will probably be around 8 hours but results could be much better than in the past as sprint trains have already exceeded past records. 
* First attempt hung during fit on epoch 17. Speculate that resource exhaustion was occuring with workers set to 8. The second run with workers=6 ran to completion. Errors from wedged runs capture during break as stall_info.txt or similar.  The first run had the slowly falling LR for stage 2, 3, 4 of mini-trainings.  During the stalled run it appeared that the model was overfitting rapidly but it also appeared that test accuracy was around 0.82. The run that ran to completion had LR reduced more aggressively, stage 2 = LR / 4, stage 3 = LR / 8, stage 4 = LR / 16. The final test error ended up being around 0.80 so it seems that the LR was too low despite this model also overfitting in the end (see chart above).
* This is the first complete model to exceed the minimum goal of 0.75 accuracy on the test set.  There's a lot of room for improvement. At this time top 10 on public leader board is 0.15234 error. Assuming the test error accurately projected to the leaderboard score (quite the assumption) then that would currently place around 60th on the LB. April 15, 2018

#### v1.6b (sprint)
<img src="/imgs/model_v1_6b.png" alt="Model v1.6b" width="800" height="400">

* _**`/src/model_firefly_v1_6.py`**_ - Fine tune model after some break in. Run for 20 total epochs of 5 rounds. First round is simply a normal mini-train with LR at 0.00025.  However for each successive mini-train unfroze the top 2 layers of Iv3 and also divided LR by 2^round.  Therefore round 2 had 2 unfrozen layers and LR of 0.000125, round 3: 4 unfrozen layers and LR of 0.0000625, etc. New record accuracy of 0.7445

#### v1.6 (sprint)
<img src="/imgs/model_v1_6.png" alt="Model v1.6" width="800" height="400">

* _**`/src/model_echidna_v1_5a.py`**_ - Fine tune model after some break in. After 20 epochs go from 172 (all) frozen Iv3 layers to 165 frozen layers. Figured out how to concatenate the model history so the chart accounts for both training regimes. New record accuracy of 0.72 and chart looks very interesting.


#### v1.5c (sprint)
<img src="/imgs/model_v1_5c.png" alt="Model v1.5c" width="800" height="400">

* _**`/src/model_echidna_v1_5.py`**_ - Adam learning rate of 0.0005 seems the best so far

#### v1.5b (sprint)
<img src="/imgs/model_v1_5b.png" alt="Model v1.5b" width="800" height="400">

* _**`/src/model_echidna_v1_5.py`**_ - Adam learning rate of 0.005 completely failed

#### v1.5a (sprint)
<img src="/imgs/model_v1_5a.png" alt="Model v1.5a" width="800" height="400">

* _**`/src/model_echidna_v1_5.py`**_ - Take model v1_3b for rapid prototyping. Experiment with tuning the optimizer to see if we can get Adam working (better) than SGD that has been used so far.  Default Adam learning rate of 0.001, seems ok.

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

#### v1.3d (sprint)
<img src="/imgs/model_v1_3d.png" alt="Model v1.3d" width="800" height="400">

* Increased epochs from 60 to 100 to give more time to converge/verify that it's over fitting.  Also changed weight initialization from default to 'he_normal'

#### v1.3c (sprint)
<img src="/imgs/model_v1_3c.png" alt="Model v1.3b" width="800" height="400">

* Increased dropout to relatively absurd level of 0.55. (But had success with that value in the past) Increased epochs to 40 because convergence should be slower since it's much more difficult to learn. This is confirmed in the above chart but it also seems like a few more epochs may show if the accuracy can improve.

#### v1.3b (sprint)
<img src="/imgs/model_v1_3b.png" alt="Model v1.3b" width="800" height="400">

* Increased dropout to relatively absurd level of 0.55. (But had success with that value in the past) Increased epochs to 40 because convergence should be slower since it's much more difficult to learn. This is confirmed in the above chart but it also seems like a few more epochs may show if the accuracy can improve.

#### v1.3a (sprint)
<img src="/imgs/model_v1_3a.png" alt="Model v1.3a" width="800" height="400">

* Tried going for 100 epochs, model seems to start overfitting at ~20 epochs. Runtime for 100 epochs was ~12 hours so that's not speeding anything up. However if runs only need to be ~20-25 epochs the runs would be just over 2.5 hours.

#### v1.3 (sprint)
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
