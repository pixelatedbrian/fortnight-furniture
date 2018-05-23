# I make all of the predictions on the images fed into me.
import keras
from keras.models import load_model

import json
import pandas as pd
import glob
import numpy as np
from sklearn.metrics import accuracy_score
from loader_bot import FireBot

if __name__ == "__main__":
    model_path = "model_v1_8_all_data_train_weights.h5"

    print("load image file paths for generators")
    # Use json to load the permanent dictionary that has been Created
    with open("../data/data_splits.json") as infile:
        data_link_dict = json.load(infile)


    # load the prediction CSV
    prediction_df = pd.read_csv("../data/sample_submission_randomlabel.csv")

    # push random values into the predictions column since there's some missing
    # images
    prediction_df.predicted.apply(lambda x: np.random.randint(1, 128))
    print(prediction_df.head())
    prediction_df.to_csv("../data/start_predictions.csv", index=False)

    print("extract files that we want")

    ##################
    ### validation ###
    ##################
    # data to use as validation to ensure that it's predicting properly
    # X_test_img_paths = data_link_dict["X_test_1"]
    # y_test = np.array(data_link_dict["y_test_1"])

    file_paths = glob.glob("../data/ready_test/*.jpg")
    file_ids = np.array([int(item.split("/")[-1].split(".")[0]) for item in file_paths])

    out_df = pd.DataFrame(file_ids, columns=["id"])

    # print("shape of y_test:", y_test.shape)

    # Parameters for Generators
    params = {'dim': (299,299),
              'batch_size': 256,
              'n_classes': 128,
              'n_channels': 3,
              'shuffle': False}

    # Generators
    predictor_generator = FireBot(file_paths, **params)

    # load the model
    print("Loading Model...")
    pred_model = load_model(model_path)
    print("Model Load Complete")

    # pred_model.compile()

    print("make predictions")
    preds = pred_model.predict_generator(generator=predictor_generator,
                                         use_multiprocessing=True,
                                         workers=6,
                                         verbose=1)

    print("supposedly have some predictions now")
    print("preds shape", preds.shape)

    print("convert preds from one-hots to integers again")
    new_preds = np.argmax(preds, axis=1)
    new_preds += 1  # indexed to 1, not 0
    print("new_preds shape:", new_preds.shape)
    print("new_preds [:20]", new_preds[:20])
    # correct = y_test == new_preds
    #
    # print("my measured accuracy is: {:2.2f}%".format(100 * np.sum(correct) / len(y_test)))
    # for model v1_8 measured acc was 93.26% (having seen all of the images before...)

    out_df.loc[:, "predicted"] = new_preds

    out_df.to_csv("../data/model_v1_8_predictions.csv", index=False)
