## fortnight-furniture

Image classification challenge competing in:
[Kaggle - Furniture Image Classification Competition](https://www.kaggle.com/c/imaterialist-challenge-furniture-2018)


#### What is the challenge?

Identify which of 128 classes an image belongs to.  The source and test images are mostly furniture and home furnishings, like an Ikea catalogue.

The training set provides links and labels for about 180k images. The scoring test set contains about 100k images.

### Overview of the modeling process for my solution.

1. **Get Data**
  * **pull_img.py** Downloaded raw image files from links using multiprocessing
  
2: **Initial EDA**
  * Looked at Distribution of Labels and Class Imbalances
  * Looked at types of images that will be trained on
  * Determine what types of data will be used and opportunities and limitations of that data
  
3: **Data Pipeline**
  * **clean_images.py Image Processing v1.0:** Open images, scale to 299x299, flip under-represented classes on vertical axis to simply augment
  * **splitter.py Stratified K-Fold Splits v1.0:** extend sklearn StratifiedKfold to generate balanced class folds to evaluate model performance. 10 folds of train 90% eval 10% generated, 3 used.
  * **loader_bot.py Data Feed Generator v1.0:** Since all images won't fit in memory made a generator that gathers and feeds batches to the model during training/evaluation
  
4. **Model**
  * Using Inception V3 transfer learning
  * Froze Iv3 layers, dropped top layer of Iv3 and added new layers for class prediction
    * Inception v3 (-1000 softmax prediction final layer)
    * 1024 FC
    * 512 FC
    * 256 FC
    * 128 softmax
   * Currently takes ~12 hours to train
    
5. **Evaluation**
  * Overall Package v0.1: 67% accuracy on 3 fold evaluation
    * Obviously not good but main goal was establishing end-to-end pipeline that can be iterated upon

#### Priorities for v0.11:
* clean_images.py Currently all images are being rescaled from whatever native resolution was to 299x299.  That is causing distortions in the image file which is most likely a problem. Easy fix too.
* Refactor model run script into a normal python script instead of jupyter notebook. The

#### Priorities for v0.2:
* Augment all images with zoom, rotation, cropping before resize to 299^2
* Improve splitter.py to not fold on augmented data, so eval data is not augmented at all (currently doesn't distinguish)
