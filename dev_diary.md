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
| 1.6b | 0.74 | 0.55 | 0.00025 | 40 | 18k | 3.7 | Four stage, use frozen Iv3 model for 5 epochs then unfreeze top 2 layers for each of remaining 3 rounds. Also decrease learning rate by LR/2^round New record accuracy 0.74  |
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
| 2.1a | 0.74 | 0.55 | 0.0005 | 40 | 18k | 5.6 | went to lighter NN, only 1024 FC layer |
| 2.1b | 0.730 | 0.55 | 0.0005 | 40 | 18k | 5.6 | model 2_1b with 512 FC layer only |
| 2.1c | 0.72 | 0.55 | 0.001 | 80 | 18k | 11.1 | model 2_1b with 140 FC layer only, SGD optimizer, lower LR, double epochs |
| 2.2a | 0.734 | 0.55 | 0.00025 | 40 | 18k | 1.7 | trying to reproduce v1.6b and establish new baseline |
| 2.2b | 0.711 | 0.55 | 0.0125 | 40 | 18k | 1.7 | swap back to SGD momentum 0.9 |
| 2.2c | 0.711 | 0.35 | 0.0125 | 40 | 18k | 1.7 | lower dropout to 0.35 to see if it gets closer to 0.74 |
| 2.2d | 0.701 | 0.45 | 0.0075 | 40 | 18k | 1.7 | raise dropout to 0.45 and lower LR to 0.0075 to see if it gets closer to 0.74 |
| 2.2e | 0.685 | 0.60 | 0.01 | 40 | 18k | 1.7 | 2.2d with ridiculously high dropout |
| 2.2f | 0.719 | 0.60 | 0.02 | 40 | 18k | 1.7 | 2.2e with LR increased to 0.02 |
| 2.2g | 0.716 | 0.50 | 0.02 | 40 | 18k | 1.7 | SGD with nesterov=True |
| 2.2h | 0.712 | 0.50 | 0.0001 | 40 | 18k | 1.7 | Back to Adam |
| 2.2i | 0.736 | 0.55 | 0.00025 | 40 | 18k | 3.1 | Back to 1.6b |
| 2.2j | 0.741 | 0.55 | 0.00025 | 40 | 18k | 3.1 | Enable augmentation without rotation, statistically insignificant sprint record accuracy |
| 2.2k | 0.734 | 0.55 | 0.00025 | 40 | 18k | 3.2 | With rotation augmentation nerfed to +-15 degrees |
| 2.2l | 0.738 | 0.55 | 0.00025 | 40 | 18k | 3.2 | With rotation augmentation nerfed to +-3 degrees |
| 2.2m | 0.713 | 0.50 | 0.0005 | 40 | 18k | 3.2 | speed up initial learning then slow down thawed learning |
| 2.2n | 0.743 | 0.55 | 0.00025 | 40 | 18k | 5.1 | Go back to 2.2l but turn on fancy_pca in image augmentation |
| 2.2n | 0.744 | 0.55 | 0.00025 | 40 | 18k | 5.1 | 2.2n with imagenet averages subtracted out of images |
| 2.2o | 0.710 | 0.55 | 0.00025 | 40 | 18k | 5.1 | Go back to 2.2o but turn on fancy_pca in image augmentation |
| 2.3a | 0.662 | 0.50 | 0.0005 | 40 | 18k | 4.0 | VGG16 model |
| 2.3b | 0.692 | 0.30 | 0.0005 | 40 | 18k | 2.7 | VGG16 model Freeze 18 layers, non-augmented |
| 2.3c | 0.680 | 0.40 | 0.00025 | 80 | 18k | 8.0 | VGG16 model Freeze 18 layers, augmented night run, increased dropout to 0.40 |
| 2.3d | 0.677 | 0.55 | 0.0005 | 40 | 18k | 2.7 | VGG16 model Freeze 18 layers, non-augmented |
| 2.3e | 0.695 | 0.55 | 0.0005 | 40 | 18k | 2.6 | VGG16 model Freeze 18 layers, non-augmented, double addon neurons |
| 2.3f | 0.680 | 0.55 | 0.0005 | 40 | 18k | 2.7 | Augmented |
| 2.4a | 0.019 | 0.55 | 0.0005 | 40 | 18k | 2.7 | ResNet 50, for some reason train and test are diverging ridiculously |
| 2.4b | 0.018 | 0.55 | 0.0005 | 40 | 18k | 2.7 | Experimented a lot, still happening |
| 2.2p | 0.746 | 0.55 | 0.00025 | 40 | 18k | 3.3 | try L1 regularization 0.00001|
| 2.2q | 0.744 | 0.55 | 0.00025 | 40 | 18k | 3.3 | try L1 regularization 0.0001|
| 2.2r | 0.741 | 0.55 | 0.00025 | 40 | 18k | 4.9 | increase first rounds epochs to 30 |
| 2.2s | 0.710 | 0.55 | 0.00025 | 40 | 18k | 3.3 | try regularized double brian layers of 30 epochs, then 10 epochs unfrozen 2 |
| 2.2t | 0.735 | 0.55 | 0.00025 | 40 | 18k | 11.7 | increase first rounds (L1 regularized) epochs to 100 |
| 2.2u | 0.750 | 0.55 | 0.00025 | 40 | 18k | 6.6 | increase first rounds epochs to 30, run 2nd minibatch 30, then last 2 bathes 10 epochs each. New record for Sprints |
| 2.2v | 0.743 | 0.55 | 0.00025 | 40 | 18k | 3.2 | image augmentation engine v3 rewrite, 4 minibatch of 10 |
| 2.2w | 0.822 | 0.55 | 0.00025 | 40 | 180k | 24.7 | normal 4 minibatch of 10 epochs with aug v3... on full 90% data |
| 2.2x | 0.744 | 0.55 | 0.00025 | 40 | 18k | 1.8 | aug v3 4x minibatch, decreased fancy_pca STD from 1.0 to 0.1 like paper advises |
| 2.2y | 0.731 | 0.55 | 0.00025 | 70 | 18k | 3.2 | aug v3 4x minibatch but R1: 20 epochs, R2: 30E, R3: 10E, R4: 10E, reduced R2-4 LR to lr/8 |
| 2.5a | 0.742 | 0.55 | 0.00025 | 40 | 18K | 1.8 | Back to normal 4 mini-batch of 10 epochs, unfreezing 2 layers after 1st mini-train. Added L2 weight decay to unfrozen layers in mini-train 2+ | 
| 2.5b | 0.742 | 0.55 | 0.00025 | 40 | 18k | 1.8 | try L2 regularization 0.0001 |
| 2.5c | 0.742 | 0.55 | 0.00025 | 40 | 18k | 1.8 | try L2 regularization 0.001 |
| 2.5d | 0.742 | 0.55 | 0.00025 | 40 | 18k | 1.8 | try L1 regularization 0.001, still not working |
| 2.5e | 0.732 | 0.55 | 0.00025 | 40 | 18k | 1.8 | seems like there is a problem of applying regularization to pretrained weights, might be able to work around, L1 reg 0.0001 |
| 2.5f | 0.736 | 0.55 | 0.00025 | 50 | 18k | 2.3 | 5 mini-train since it's learning more slowly with the regularization |
| 2.5g | 0.736 | 0.55 | 0.00025 | 50 | 18k | 2.3 | switch back to L2 regularization at 0.0001, oddly seems the same |
| 2.5h | 0.748 | 0.55 | 0.00025 | 100 | 18k | 5.0 | night train, increase reg to 0.0005, 10 mini-train |
| 2.5i | 0.741 | 0.55 | 0.00025 | 100 | 18k | 5.0 | Make LR stable after mini-train 2 and on |
| 2.5j | 0.743 | 0.55 | 0.0005 | 100 | 18k | 5.0 | undid constant LR, increased reg, increased LR, tried plot fix |
| 2.5k | 0.742 | 0.55 | 0.0005 | 100 | 18k | 5.0 | fix zeros in plots in plot function, increase regularization to 0.01 |
| 2.5l | 0.725 | 0.55 | 0.0001 | 150 | 18k | 7.4 | switch regularization back to L2, still high value of 0.01, reduce LR, increase minitrains |
| 2.5m | 0.760 | 0.50 | 0.00025 | 150 | 18k | 7.4 | 0.7596 accuracy, new record. Not regularizing again though? increase LR back to 0.00025, decrease per minitrain drop of LR to 1.5^MT, reduce dropout slightly to 0.50 |
| 2.5n | ? | 0.55 | 0.00025 | 150 | 18k | 7.4 | --Many changes, see list below-- Crashed in run probably from memory exhaustion |
| 3.0a | 0.515 | 0.0 | 0.0025 | 50 | 18k | 3.7 | ResNet50 from scratch, needs dropout, prob doesn't need adaptive LR right now |
| 3.0b | 0.440 | 0.5 | 0.0025 | 50 | 18k | 3.7 | added in dropout, made LR drop by LR / (minitrain + 1) |
| 3.0c | 0.520 | 0.05 | 0.0025 | 50 | 18k | 3.7 | reduced dropout |
| 3.0d | 0.708 | 0.25 | 0.0025 | 50 | 180k | 3.7 | warm_start with weights from 3.0c, did it on full image set 180k images about 1.33 hr per epoch, seems to need more dropout |
| 2.6a | 0.53 | 0.55 | 0.00025 | 60 | 18k | 2.7 | Trying to reestablish v2.5 series baseline but failing |
| 2.6b | 0.57 | 0.55 | 0.00025 | 30 | 18k | 1.5 | Still failing |
| 2.6c | 0.72 | 0.55 | 0.00025 | 150 | 18k |7.5 | Finally got it, it was because the corpus means were being subtracted during image processing, instead of the Imagenet means |
| 2.6d | 0.755 | 0.55 | 0.00025 | 100 | 18k | 5.0 | Concatenating Raven model/data. See notes. |
| 2.6e | 0.750 | 0.55 | 0.00025 | 30 | 18k | 1.5 | Removed softmax from Raven model, extended non-fine tune 1st mini-train to 20 epochs |
| 2.6f | 0.753 | 0.62 | 0.00025 | 100 | 18k | 5.0 | Concatenating Raven model/data. See notes. |
| 2.6g | 0.760 | 0.55 | 0.00025 | 100 | 18k | 5.0 | Loading Raven weights, freezing Raven model |
| 2.6h | 0.755 | 0.55 | 0.00025 | 150 | 18k | 7.5 | 150 epochs night run |
| 2.6i | 0.827 | 0.55 | 0.00025 | 40 | 180k | 13.4 | Run on full training data, new accuracy record, old loader_bot augmentation |
| 2.6j | 0.832 | 0.55 | 0.00025 | 40 | 180k | 14.27 | Learning Rate Annealing, Run on full training data, new accuracy record at epoch 29 |
| 2.7a | 0.722 | 0.55 | 0.00025 | 175 | 18k | 7.0 | 175 epochs night run, full 20 count static augmentation, aug odds 100% |
| 2.7b | 0.709 | 0.55 | 0.00025 | 150 | 18k | 6.4 | 150 epochs, increasing aug odds from 0% to 100% by 5% increments |
| 2.7c | n/a | 0.55 | 0.00025 | 150 | 18k | 6.4 | loaded weights from 2.5m, Train/test folds are different so the weights have seen some of the test images. Pulling score |
| 2.7d | 0.693 | 0.55 | 0.00025 | 150 | 18k | 6.4 | no augmentation run to reestablish baseline and pre-train weights on correct train/test splits  |
| 2.7e | 0.735 | 0.50 | 0.00025 | 150 | 18k | 6.0 | 150 epochs, no augmentation, reduce regularization some dropout to 0.50 |
| 2.7f | 0.738 | 0.50 | 0.00025 | 150 | 18k | 6.0 | import weights from 2.7f and do full augmentation on sprint set |
| 2.7g | 0.751 | 0.50 | 0.00025 | 200 | 18k | 8.6 | full augmentation, less regularization DO 0.50 200 total epochs, LR = LR / 1.5^mini-train |
| 2.7h | 0.823 | 0.55 | 0.00025 | 200 | 180k | 5.6 | full train, full augmentation, use weights from 2.7g |
| 2.7i | 0.830 | 0.55 | 0.00025 | 30 | 180k | 5.5 | same as 2.7h but with learning rate annealing |
| 2.7j | 0.748 | 0.55 | 0.00025 | 60 | 18k | 8.6 | base 2.7 but with extra layer in Brian DNN making it 4 layers 2048 FC -> 1024 FC -> 512 FC -> 256 FC -> 128 Softmax |
| 2.7k | 0.727 | 0.575 | 0.00025 | 25 | 18k | 8.6 | 2.7j with heavy regularization, L2 0.075, killed train early |
| 2.7l | 0.750 | 0.625 | 0.00025 | 25 | 18k | 8.6 | 2.7j with very heavy regularization, L2 0.125 |

#### v2.7l
<img src="/imgs/model_v2_7l.png" alt="Model v2_7l" width="800" height="400">

* `/src/model_hawk_v2_7.py`
* Tried using a deeper net, it definitely learned faster but variance is huge as the network overfits yet the validation error again plateaus at ~0.75.  At this point seems like inherent limitations in Inception v3

#### v2.7a
<img src="/imgs/model_v2_7a.png" alt="Model v2_7a" width="800" height="400">

* `/src/model_hawk_v2_7.py`
* Extension of 2.5
* Removed the Raven DNN 'assist' because it was preventing static augmentation from working
* got static augmentation working but it doesn't seem to be helping

### Static Augmentation of Images
 * 

### Submitted Results
 * Scored slightly worse on the leaderboard than before.  Accuracy was worse by 1% to 20%
 * Found that the color channels of the images are inconsistent between training sets which is messing things up
 * The static image augmentation is failing to progress past 2% accuracy, which is infuriating.  Not sure what is going on.
 * For static image aug fixed it so that it would be maintaining RBG in a standard manner. The main cause of the problem is that CV2 opens images in a non RGB format X-(
 * Going to attempt to disable smart crop for the static image augmentation and see if that helps.

#### v2.6j
<img src="/imgs/model_v2_6j.png" alt="Model v2_6j" width="800" height="400">

* `/src/model_gyrfalcon_v2_6.py`
* Extension of 2.6i
* New record accuracy on validation set for full train run 0.832 at E29
* Original 'flawed' loader_bot augmentation (of already cropped images)
* L2 regularization slightly decreased from 0.05 to 0.025
* Allow Raven to thaw after 5 epochs for 'fine tuning'
* Learning rate annealing for period of 'bump' around every 7 epochs (40E total)

#### v2.6i
<img src="/imgs/model_v2_6i.png" alt="Model v2_6i" width="800" height="400">

* `/src/model_gyrfalcon_v2_6.py`
* Loading in pretrained weights from a 9% validation accuracy Raven model
* New record accuracy on validation set for full train run
* L2 regularization slightly decreased from 0.05 to 0.025

### Found a sort of 'bug' in what images were being fed into the augmentation system.
 * `/src/loader_bot_omega.py`
 * LoaderBot was loading in images from `root/data/stage3_imgs/` 
 * I thought that those images were cleaned up images (1x1 images and similar removed) but otherwise full images
 * Upon switching LoaderBot to use the original raw images from `root/data/imgs/` I found that training time went from ~3 min per epoch to ~15-18min per epoch. 
 * For a long time I thought that there was some kind of bug in LoaderBot but finally I loaded some `stage3_imgs` files and saw that they were all already 299x299px center crops of the raw images.  Which means that all previous augmentation was being performed on the small central crops as opposed to trying to take broader or zoomed slices of the original image.
 * Given that correctly augmenting was taking 15+ min per epoch (on the Sprint data no less, which is only 10% of the training set) that made it look like training on the whole set was going to be incredibly slow.
 * Therefore I finally implemented the long considered 'static augmentation'.  Essentially each raw image is loaded, a FOR LOOP is run and the image is augmented x amount of times. (20 in this case)  Then those images are saved with the incremental number added to the file name.
 * `/src/loader_bot_reloaded.py` was created
 * LoaderBot - Reloaded has an option to randomly load from the specified range. So now it will look at the file path specified by the training data dictionary. Then, given the range of options (0 to 19 in this case), it will randomly select a pre-augmented file to be used for this particular batch.
 * This will hopefully provide a combination of solid augmentation and also speed up epochs compared to previous versions as there is less processing of the images involved.


#### v2.6f (sprint)
<img src="/imgs/model_v2_6f.png" alt="Model v2_6f" width="800" height="400">

* `/src/model_gyrfalcon_v2_6.py`
* Loading in pretrained weights from a 9% validation accuracy Raven model
* Increased dropout to 0.62, didn't really seem to help the overfitting, at least within 100 total epochs
* Kind of concerned that backprop is 'ruining' the 'better' pretrained weights of the Raven model
* Next version going to try freezing Raven model, at least initially, in similar concept to Iv3

#### v2.6e (sprint)
<img src="/imgs/model_v2_6e.png" alt="Model v2_6e" width="800" height="400">

* `/src/model_gyrfalcon_v2_6.py`
* The same as 2.6d except that the softmax layer of Raven DNN has been removed.
* Performance slightly worse
* Tried extending the 1st mini-train without fine tuning of Inception v3 enabled in order to give the Raven DNN more time to train. Didn't seem to help

#### v2.6d (sprint)
<img src="/imgs/model_v2_6d.png" alt="Model v2_6d" width="800" height="400">

* `/src/model_gyrfalcon_v2_6.py`
* Concatenated a small DNN side by side with the Inception v3 model.  The Raven DNN uses the image meta features listed below to attempt to make predictions
* When training on image meta data accuracy was able to get to about 10% on the validation set. Not great by any means, but it does seem to imply some signal as it's about 12x the accuracy one would expect from random guessing.
* First training of the dual models involved untrained weights for the Raven DNN
* After the fact also realized that when the Raven DNN was being concatenated that it still had the softmax predictions layer left over from it's solo-development stage.  Therefore the 'Brian net' DNN after InceptionV3/Raven was taking in probablities directly from the Raven model.
* Got back to 75% accuracy which is above average for sprints but still not an improvement

#### v2.6a-c (sprint)
<img src="/imgs/model_v2_6a.png" alt="Model v2_6a" width="800" height="400">

* `/src/model_gyrfalcon_v2_6.py`
* The goal was to concatenate a DNN that works on image meta features with the relatively successful v2.5 model framework using Sprint data sets
* First simply tried to prove that the baseline had similar performance to the original v2.5 series
* This was not the case, performance was crushed from ~75% accuracy down to low 50%s.
* Eventually figured out that the ResNet train on only this image corpus (not transfer learning) averages were being subtracted instead of the ImageNet averages.  This caused the massive drop in accuracy.  Lesson learned _**Subtracting the correct means from image color channels in pre-processing is extremely important.**_

### Image files have sort of meta features that can be inferred from the image itself. 
 * height
 * width
 * aspect ratio of w/h
 * file size (byes)
 * number of pixels (h * w)
 * pixels / byte of file (proxy for how compressible the image is?)
 * color_channel_mean of file (r, g, b)
 * accumulated error per color (resididual data from finding image set global standard deviation, see more details of that below)
 
The above data was fed into a basic deep neural network. Randomly attempting to guess a image class should be about 0.78% accurate.  The DNN, using the meta features above, was about to get ~10% accuracy on the test set.  Obviously not amazing _but_ it does seem to imply that there is signal there. Newest idea is to concatenate this data into an InceptionV3 transfer learning model to try to give the model a bit more information.  I believe that this complies with the rules as the meta features all come from the image data itself and not the filename or url at all.

Going to try to extend 2.5x to 2.6 with data concatenation.


#### v3.0a (sprint)
<img src="/imgs/model_v3_0a.png" alt="Model v3_0a" width="800" height="400">

* `/src/model_acidmaw_3_0.py`
* try training a brand new ResNet50
* use SGD with Nesterov
* currently no dropout or other forms of explicit regularization (still augmenting images though)
* crashed on batch size 128, working on BS=16, going to try higher values as might speed things up
* BS=64 crashed, BS=32 seems stable so far
* train 0.0709 at end of 3rd epoch first tune
* modified LR from Adam-like 0.00025 to SGD-like 0.0025
* train 0.1291 at end of 3rd epoch after 2nd non-full run tune

### Arranging deck chairs on the Titanic?  
 * Been tweaking Inceptionv3 for over a month now
 * Tried VGG16 and that didn't seem to do well
 * Tried ResNet50 and that seemed to generalize to train very well but due to a bug (or something) it couldn't predict on the test set even (predictions essentially were random, validation accuracy never got above 1.5%)
 * Going to try to train a new ResNet50 on this data for 3.0

#### v2.5o (sprint)
<img src="/imgs/model_v2_5o.png" alt="Model v2_5o" width="800" height="400">

* `/src/model_falcon_v2_5.py`
* try switching to SGD again

#### v2.5n (sprint)
<img src="/imgs/model_v2_5n.png" alt="Model v2_5n" width="800" height="400">

* `/src/model_falcon_v2_5.py`
* increase dropout back to 0.55
* change learning rate to programmed annealing
* mini-trains x 2  (30) but epochs per MT reduced from 10 to 5 to have more granular control
* increase L2 regularization from 0.01 to 0.05
* cap layer thaw to 50
* reduce thaw from 5 per mini-train to int(2.5 per minitrain)
* crashed out, out of memory on GPU

#### v2.5m (sprint)
<img src="/imgs/model_v2_5m.png" alt="Model v2_5m" width="800" height="400">

* `/src/model_falcon_v2_5.py`
* **finally** seems regularized, but still plateauing at ~0.73 accuracy on test set
* increase learning rate slightly
* modify LR decay to be less aggressive since mini-trains are getting higher (15 now)
* changed LR decay of minitrain from LR / 2**(MT-1) -> LR / 1.5**(MT-1)
* reduce dropout to 0.50 since that seems to be pretty standard in papers

#### v2.5l (sprint)
<img src="/imgs/model_v2_5l.png" alt="Model v2_5l" width="800" height="400">

* `/src/model_falcon_v2_5.py`
* increase regularization to 0.01 (seems high but we'll see)

#### v2.5k (sprint)
<img src="/imgs/model_v2_5k.png" alt="Model v2_5k" width="800" height="400">

* `/src/model_falcon_v2_5.py`
* increase regularization to 0.01 (seems high but we'll see)

#### v2.5j (sprint)
<img src="/imgs/model_v2_5j.png" alt="Model v2_5j" width="800" height="400">

* `/src/model_falcon_v2_5.py`
* undo the constant LR back to LR / 2^MT
* change layer thawing from 5 per mini-train back to 2 per mini-train
* with more mini-trains the unfrozen layers will still ultimately be a lot more (from 8 to 20)
* attempt to fix the history weirdness, graph can't be worse than last time
* increase the L2 regularization from 0.0005 to 0.001
* increase the starting LR from 0.00025 to 0.0005
* the model diverges again at later training


#### v2.5i (sprint)
<img src="/imgs/model_v2_5i.png" alt="Model v2_5i" width="800" height="400">

* `/src/model_falcon_v2_5.py`
* 10 mini-trains of 10 epochs
* tried fixing the zero padding on the histories, made it worse
* constant learning rate of mini-trains after MT1 of LR/10, seems like might be too low
* regularization is failing, model is diverging widely again

#### v2.5h (sprint)
<img src="/imgs/model_v2_5h.png" alt="Model v2_5h" width="800" height="400">

* `/src/model_falcon_v2_5.py`
* 10 mini-trains of 10 epochs
* 0.748 so showing some promise
* had to reduce batch size to 128 but doesn't seem to be affecting (already slow) speed
* But the LR is currently LR / 2**(mini-train-1) so by mini-train 9 the LR is tiny
* for v2.5i make LR / 10 constant after the 1st mini-train
* history plot is messed up, seems like saving the model and loading is corrupting the history somehow

#### v2.5g (sprint)
<img src="/imgs/model_v2_5g.png" alt="Model v2_5g" width="800" height="400">

* `/src/model_falcon_v2_5.py`
* went back to L2 regularization
* results the same... because regularization still isn't working

#### v2.5f (sprint)
<img src="/imgs/model_v2_5f.png" alt="Model v2_5f" width="800" height="400">

* `/src/model_falcon_v2_5.py`
* Try going 5 mini-train runs

#### v2.5e (sprint)
<img src="/imgs/model_v2_5e.png" alt="Model v2_5e" width="800" height="400">

* `/src/model_falcon_v2_5.py`
* Saving and loading seems to finally get the regularization working
* Train accuracy went from 95% to 84%, test accuracy didn't increase but might just need more time now?

#### v2.5d (sprint)
<img src="/imgs/model_v2_5d.png" alt="Model v2_5d" width="800" height="400">

* `/src/model_falcon_v2_5.py`
* Tried switching to L1 regularization
* still not working

#### v2.5c (sprint)
<img src="/imgs/model_v2_5c.png" alt="Model v2_5c" width="800" height="400">

* `/src/model_falcon_v2_5.py`
* Increased L1 regularization value of 0.001
* same, looks like regularization isn't working

#### v2.5b (sprint)
<img src="/imgs/model_v2_5b.png" alt="Model v2_5b" width="800" height="400">

* `/src/model_falcon_v2_5.py`
* Increased L2 regularization value of 0.0001
* same, looks like regularization isn't working

#### v2.5a (sprint)
<img src="/imgs/model_v2_5a.png" alt="Model v2_5a" width="800" height="400">

* `/src/model_falcon_v2_5.py`
* Started out by adding L2 regularization to the thawed layers
* Initial L2 regularization value of 0.00001 used
* Didn't do much

### Looking at the charts the main problem is still overfitting
 * The regularization on the 'brian layers' seems to have helped with the early mini-train overfitting issues
 * There are still really bad divergences of train/test when the Inception layers start to get thawed.
 * Searched for, and found, a way to add regularization to existing layers
 * new version is model v2.5 which has regularization in the thawed layers as well
 * starting out trying L2 regularization and then may try L1 since it helped on my added layers

#### v2.2y (sprint)
<img src="/imgs/model_v2_2y.png" alt="Model v2_2y" width="800" height="400">

* `/src/model_cactuswren_v2_2.py`  
* modified epochs for each training
* mini-train 1: 20 epochs, mini-train 2: 30 epochs, mini-train 3: 10 epochs, mini-train 4: 10 epochs
* reduced Learning Rate on all unfrozen mini-trains (2-4) to constant LR / 8 instead of LR / 2 -> LR / 4 -> LR / 8
* Seems worse

#### v2.2x (sprint)
<img src="/imgs/model_v2_2x.png" alt="Model v2_2x" width="800" height="400">

* `/src/model_cactuswren_v2_2.py`  
* sprint with 4x mini-train
* decreased fancy_pca standard deviation from 1.0 to 0.1 which is what the paper suggests
* eh, still about the same

#### v2.2w (FULL)
<img src="/imgs/model_v2_2w.png" alt="Model v2_2w" width="800" height="400">

* `/src/model_cactuswren_v2_2.py`  
* Tried augmentation v3 on full data
* Used typical mini-train 4 / epochs 10 setup
* About the same score as the past but took over 24 hours (ouch)

### Image Augmentation 3 rewrite
* tuned fancy_pca to be more standardized and readable
* modified the crop alg so that it will crop from more of a picture's available data instead of being centered in the middle
* modified the crop alg so that it doesn't zoom as much, so more of the central subject is in frame
* broke out image augmentation into it's own library so that loader_bot_omega can simply import what it needs
* tested and validated new principles in jupyter notebook data_augmentation_v3.ipynb

#### v2.2v (sprint)
<img src="/imgs/model_v2_2v.png" alt="Model v2_2v" width="800" height="400">

* `/src/model_cactuswren_v2_2.py`  
* Image augmentation v3 rewrite
* Went back to typical sprint: 4 mini-train of 10 epochs with LR per mini-train being LR, LR / 2, LR / 4, LR / 8
* Didn't seem to move the needle

#### v2.2u (sprint)
<img src="/imgs/model_v2_2u.png" alt="Model v2_2u" width="800" height="400">

* `/src/model_cactuswren_v2_2.py`  
* mini-train 1: 30 epochs, mini-train 2: 30 epochs, mini-train 3-4: 10 epochs
* new sprint record of 0.75

#### v2.2t (sprint)
<img src="/imgs/model_v2_2t.png" alt="Model v2_2t" width="800" height="400">

* `/src/model_cactuswren_v2_2.py`  
* Night run, increased the mini-train 1 epochs to 100
* Didn't really help

#### v2.2s (sprint)
<img src="/imgs/model_v2_2s.png" alt="Model v2_2s" width="800" height="400">

* `/src/model_cactuswren_v2_2.py`  
* Try regularized 'double brian layers' for 30 epochs
* Then 10 epochs of the 2/4/6 unfrozen layers
* Seemed worse

#### v2.2r (sprint)
<img src="/imgs/model_v2_2r.png" alt="Model v2_2r" width="800" height="400">

* `/src/model_cactuswren_v2_2.py`  
* Increase first mini-train epochs to 30 to see if that helped (not really)

#### v2.2q (sprint)
<img src="/imgs/model_v2_2q.png" alt="Model v2_2q" width="800" height="400">

* `/src/model_cactuswren_v2_2.py`  
* Increased L1 regularization (0.0001) on 'brian layers'
* Now seeing some good regularizing effect on 'brian layer' training but not on unfrozen layers

#### v2.2p (sprint)
<img src="/imgs/model_v2_2p.png" alt="Model v2_2p" width="800" height="400">

* `/src/model_cactuswren_v2_2.py`  
* L1 regularization (0.00001) on 'brian layers'
* Didn't really see regularizing effect

#### v2.4c(print)
<img src="/imgs/model_v2_4c.png" alt="Model v2_4c" width="800" height="400">

* `/src/model_eagle_v2_4.py`  
* Going to fit a model for ~5 epochs, then use FireBot to make predictions on the same data that was used to train.  The error should be similar, if not the same. If the accuracy is still ~1% then almost certainly something is wrong with the predictions coming out of the model.

#### v2.4b(sprint)
<img src="/imgs/model_v2_4b.png" alt="Model v2_4b" width="800" height="400">

* `/src/model_eagle_v2_4.py`  
* Pretty convinced that something is going very wrong with the predict part of this model.
* Next run I will try to predict on the train data and evaluate the error of the predictions

#### v2.4a(sprint)
<img src="/imgs/model_v2_4a.png" alt="Model v2_4a" width="800" height="400">

* `/src/model_eagle_v2_4.py`  
* Implemented ResNet 50 but it is going horribly wrong and for some reason predictions are essentially random
* I don't think that it is an over fitting problem because it's consistent even when train accuracy is very low too
* Model seems like it would have more potential than VGG16 but predictions are totally messed up

#### v2.3f(sprint)
<img src="/imgs/model_v2_3f.png" alt="Model v2_3f" width="800" height="400">

* `/src/model_duck_v2_3.py`  
* 2.3e with augmentation

#### v2.3e(sprint)
<img src="/imgs/model_v2_3e.png" alt="Model v2_3e" width="800" height="400">

* `/src/model_duck_v2_3.py`  
* seems to be having difficulty generalizing and real VGG FC layers are quite large (4096 iirc)
* so increasing my FC layers to 2048 -> 1024 -> 512 to see if it helps
* running non-augmented to save some time

#### v2.3d(sprint)
<img src="/imgs/model_v2_3d.png" alt="Model v2_3d" width="800" height="400">

* `/src/model_duck_v2_3.py`  
* no augmentation
* 40 epochs
* increased dropout back to 0.55

#### v2.3c(sprint)
<img src="/imgs/model_v2_3c.png" alt="Model v2_3c" width="800" height="400">

* `/src/model_duck_v2_3.py`  
* enabled augmentation
* double epochs
* increased dropout

#### v2.3b(sprint)
<img src="/imgs/model_v2_3b.png" alt="Model v2_3b" width="800" height="400">

* `/src/model_duck_v2_3.py`  
* disabled augmentation
* decreased dropout

#### v2.3a(sprint)
<img src="/imgs/model_v2_3a.png" alt="Model v2_3a" width="800" height="400">

* `/src/model_duck_v2_3.py`  
* VGG16 model first run
* Try to start unfreezing layers after mini-train 1, starting with 18th layer

#### v2.2p(sprint)
<img src="/imgs/model_v2_2o.png" alt="Model v2_2o" width="800" height="400">

* `/src/model_cactuswren_v2_2.py`  
* Re-enable fancy_pca after working through some LoaderBot improvements.

#### v2.2o(sprint)
<img src="/imgs/model_v2_2o.png" alt="Model v2_2o" width="800" height="400">

* `/src/model_cactuswren_v2_2.py`  
* Subtract imagenet averages from images

#### v2.2n(sprint)
<img src="/imgs/model_v2_2n.png" alt="Model v2_2n" width="800" height="400">

* `/src/model_cactuswren_v2_2.py`  
* seems like 2.2m might have eventually gotten a better score
* but going back to 2.2m and enabling fancy_pca as supposedly I have that working now
* have the standard deviation set at 100 but will turn down to 10 for inaugaural run, also will run on every image

#### v2.2m(sprint)
<img src="/imgs/model_v2_2m.png" alt="Model v2_2m" width="800" height="400">

* `/src/model_cactuswren_v2_2.py`  
* speed up initial learning but slow down later epochs, last one before dropping some upper layers and see if it doesn't overfit as much

#### v2.2l(sprint)
<img src="/imgs/model_v2_2l.png" alt="Model v2_2l" width="800" height="400">

* `/src/model_cactuswren_v2_2.py`  
* enable current rotation with augmentation, of est range +-3degrees and rotation occurs 50% of the time

#### v2.2k (sprint)
<img src="/imgs/model_v2_2k.png" alt="Model v2_2k" width="800" height="400">

* `/src/model_cactuswren_v2_2.py`  
* enable current rotation with augmentation, of est range +-15degrees and rotation occurs 50% of the time

#### v2.2j (sprint)
<img src="/imgs/model_v2_2j.png" alt="Model v2_2j" width="800" height="400">

* `/src/model_cactuswren_v2_2.py`  
* image augmenation enabled without rotation

#### v2.2i (sprint)
<img src="/imgs/model_v2_2i.png" alt="Model v2_2i" width="800" height="400">

* `/src/model_cactuswren_v2_2.py`  
* go back to model 1.6b, lr=0.0025, dropout=0.55


#### v2.2h (sprint)
<img src="/imgs/model_v2_2h.png" alt="Model v2_2h" width="800" height="400">

* `/src/model_cactuswren_v2_2.py`  
* back to Adam


#### v2.2g (sprint)
<img src="/imgs/model_v2_2g.png" alt="Model v2_2g" width="800" height="400">

* `/src/model_cactuswren_v2_2.py`  
* enable nesterov momentum
* going back to Adam if this doesn't help

#### v2.2f (sprint)
<img src="/imgs/model_v2_2f.png" alt="Model v2_2f" width="800" height="400">

* `/src/model_cactuswren_v2_2.py`  
* double learning rate to 0.02

#### v2.2d (sprint)
<img src="/imgs/model_v2_2d.png" alt="Model v2_2d" width="800" height="400">

* `/src/model_cactuswren_v2_2.py`  
* raise dropout again
* decrease learning rate

#### v2.2d (sprint)
<img src="/imgs/model_v2_2d.png" alt="Model v2_2d" width="800" height="400">

* `/src/model_cactuswren_v2_2.py`  
* raise dropout again
* decrease learning rate

#### v2.2c (sprint)
<img src="/imgs/model_v2_2b.png" alt="Model v2_2b" width="800" height="400">

* `/src/model_cactuswren_v2_2.py`  
* lower dropout to see if it lets us get closer to 0.74
* graph of 2.2b seems to indicate test acc was close to having plateaued
* not doing weight decay yet

#### v2.2b (sprint)
<img src="/imgs/model_v2_2b.png" alt="Model v2_2b" width="800" height="400">

* `/src/model_cactuswren_v2_2.py`  
* change optimizer to SGD learning rate (from models 1.3-1.4) of 0.0125, re-use momentum of 0.9
* not doing weight decay yet

#### v2.2a (sprint)
<img src="/imgs/o_model_v2_2a.png" alt="Model v2_2a" width="800" height="400">

* `/src/model_cactuswren_v2_2.py`  
* Back to 1024FC -> 512FC -> 256FC -> 128 softmax 'brian net'
* Roll everything back to model v1.6b
  * Adam optimizer
  * LoaderBot v1.0
  * Loading files from drive, no augmentation
  * 4 sets of 10 epochs with 2 layers thawed and LR / 2^mini-train
* Now logging history data of model to make comparisons easier (JSON files in /src/histories/)
* Refactored model methods to have optimizer as a parameter so it's easier to modify that
* Refactored save files stuff so now only need to modify one model info parameter instead of multiple strings for saving
  
## Having problems
* Currently it's unclear that image augmentation is helping at all. Despite different attempts on sprint models still have not exceeded the past sprint validation accuracy of ~74%.
* But augmented models are taking 2-3x as long to train so currently augmentation is increasing computation expense massively with no discernable gain.
* Trying to revise network structure hasn't been helping, either. (2.1x models)
* Haven't yet tried evaluating a network on the full set but if the sprints aren't improving then I'm unconvinced that it would be worth spending that much time trying to evaluate that.

### Going forward (Immediate):
* Roll back to 1.6 for speed, as model v2.2x
  * 2.2a: Test with non-augmented data, get a baseline for 2.2
  * 2.2b: Evaluate SGD and see how it compares to baseline
  * 2.2c: adjust dropout lower to see if it gets closer to previous 1.6 score with SGD
  * 2.2f: Evaluate weight decay for SGD and see how it compares to baseline
  * 2.2e: Taking better of 2.2a/b feed in current state of augmented images
  * 2.2f: Disable rotation in augmentation and evaluate
  
### Going forward (week+ time range):
* Reading about Yelp image classification winner they mentioned that dropping the last layer only was problematic because Inception v3 weights were overfitted to the 1000 class Iv3 predictors. So that person picked up features from the last Average Pooling and then worked from there.
* In that case the person used an ensemble of image models (vgg, Inception, and ResNet iirc) and also used PCA on the features that came out of the models.
* Should definitely evaluate other models, just to see how they perform on this problem.
  * So prob try like VGG16 and ResNet
  * Try dropping a few of the last layers and then using my own layers for the final predictions
  * I've thought about augmenting (crop) test images a bit and then finding the average class prediction and using that as the final prediction. This seems to have been part of the winning Yelp image classification solution.  Appealing that it's not as computationally intensive during train.
  * Can try to ensemble predictions from other models but then that also creates computational complexity problems as well as just being a pain in pipeline.
  

#### v2.1c (sprint)
<img src="/imgs/model_v2_1c.png" alt="Model v2_1c" width="800" height="400">

  * 140 neuron FC and still overfitting, esp after unfreezing some layers.
  * Reading literature it seems like SGD optimizer is thought to perform better for image classification
  * Doubled epochs and decreased LR to 0.001

#### v2.1b (sprint)
<img src="/imgs/model_v2_1b.png" alt="Model v2_1b" width="800" height="400">

  * Go down to 512FC
  * Still didn't help with overfitting, mainly because unfreezing allows it to overfit rapidly.
  * Reading some people's opinions online after unfreezing dropping LR by an order of magnitude is suggested
  * have tried that but still overfitting even though dropout is at 0.55 too
  * Augmentation still doesn't seem to be helping, also

#### v2.1a (sprint)
<img src="/imgs/model_v2_1a.png" alt="Model v2_1a" width="800" height="400">

  * Previous models used Inception v3 and dropped the softmax.  Then added a new fully connected backend of 1024FC -> 512FC -> 256FC -> 128 Softmax
  * Right now a large concern is how much the model ends up overfitting as far as the train set relative to the test set.
  * So try reducing the network in front of the softmax to try to reduce the overfitting.
  * For model 2.1 only use a 1024FC layer before 128 class softmax
  * Didn't seem to help, error on sprint is still 0.74 and still overfitting like crazy later in higher epochs

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
