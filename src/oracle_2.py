# I make all of the predictions on the images fed into me.
import keras
from keras.models import load_model

import json
import pandas as pd
import glob
import numpy as np
from sklearn.metrics import accuracy_score
from loader_bot_reloaded import FireBot


if __name__ == "__main__":
    model_path = "weights/model_v2_6_full_weights_0_830.h5"

    print("load image file paths for generators")

    # # try to test on test split first to ensure that things are working
    # # Use json to load the permanent dictionary that has been Created
    # with open("../data/metadata_splits.json") as infile:
    #     data_link_dict = json.load(infile)
    #
    # # convert to dataframe with specific order so we know path is in index 0
    # test_data = pd.DataFrame(data_link_dict["X_test_1"],
    #                          columns=['path', 'aspect_ratio', 'b_accum_err',
    #                                   'b_mean', 'file_size', 'g_accum_err',
    #                                   'g_mean', 'h', 'pix_pb', 'pixels',
    #                                   'r_accum_err', 'r_mean', 'w'])
    #
    # test_data = test_data.values
    #
    # y_test = data_link_dict["y_test_1"]

    # Use json to load the permanent dictionary that has been Created
    with open("../data/metadata_lb_validation.json") as infile:
        data_dict = json.load(infile)

    # convert to dataframe with specific order so we know path is in index 0
    test_data = pd.DataFrame(data_dict, columns=['path', 'aspect_ratio',
                                                 'b_accum_err', 'b_mean',
                                                 'file_size', 'g_accum_err',
                                                 'g_mean', 'h', 'pix_pb',
                                                 'pixels', 'r_accum_err',
                                                 'r_mean', 'w'])
    test_data = test_data.values
    # test_data = test_data.values[:, 0]

    print("extract files that we want")

    ##################
    ### validation ###
    ##################
    # data to use as validation to ensure that it's predicting properly
    # X_test_img_paths = data_link_dict["X_test_1"]
    # y_test = np.array(data_link_dict["y_test_1"])

    # print("shape of y_test:", y_test.shape)

    # DO = 0.25

    # Parameters for Generators
    params = {'dim': (299, 299),
              'batch_size': 32,
              'n_classes': 128,
              'n_channels': 3,
              'augment': True,
              'Raven': True,
              'shuffle': False}

    # print("last test data:\n\n")
    #
    # for item in test_data[-15:]:
    #     print(item, "\n")
    #
    # print("\n\n")

    # Generators
    predictor_generator = FireBot(test_data, **params)

    # print("prebuild model to load weights into")
    # # setup model
    # # include_top=False excludes final FC layer
    # inception = InceptionV3(weights='imagenet', include_top=False)
    #
    # # DNN for meta-image features with optional transfer learning
    # raven_model = Raven(0.05, classes=128, load_weights=True)
    #
    # model = meta_brian_layers(inception, raven_model, 128, DO, l1_reg=0.0001)

    # load the model
    print("Loading Model...")
    model = load_model(model_path)
    # model.load_weights(model_path)
    print("Model Load Complete\n\n")

    # pred_model.summary()

    print("\n\n")

    # pred_model.compile()

    print("make predictions\n\n")
    preds = model.predict_generator(generator=predictor_generator,
                                    use_multiprocessing=False,
                                    verbose=1)

    print("supposedly have some predictions now\n\n")
    print("preds shape\n\n", preds.shape)

    print("convert preds from one-hots to integers again\n\n")
    new_preds = np.argmax(preds, axis=1)
    new_preds += 1  # indexed to 1, not 0
    print("new_preds shape:", new_preds.shape)
    print("new_preds [:20]", new_preds[:20])

    # load the prediction CSV
    prediction_df = pd.read_csv("../data/sample_submission_randomlabel.csv")

    # push random values into the predictions column since there's some missing
    # images
    # instead of random just push the most common class, 20
    # prediction_df.predicted.apply(lambda x: np.random.randint(1, 128))
    prediction_df.predicted = 20

    # print(prediction_df.head())
    # prediction_df.to_csv("../data/start_predictions.csv", index=False)

    # correct = y_test == new_preds

    # print("my measured accuracy is: {:2.2f}%".format(100 * np.sum(correct) / len(y_test)))
    # for model v1_8 measured acc was 93.26% (having seen all of the images before...)
    # for model v2_6i measured acc was 83.6% which during training it was around 83.2%
    # file_ids = np.array([int(item.split("/")[-1].split(".")[0]) for item in test_data])
    file_ids = np.array([int(item[0].split("/")[-1].split(".")[0]) for item in test_data])

    # raw_preds = pd.DataFrame(preds)
    # raw_preds.loc[:, "id"] = file_ids

    # raw_preds.to_csv("../data/m2_6i_cfix_raw_preds4.csv", index=False)

    out_df = pd.DataFrame(file_ids, columns=["id"])
    # Save results for submission
    out_df.loc[:, "predicted"] = new_preds

    out_df.to_csv("../data/model_v2_6i_raw_cfix_predictions.csv", index=False)

    result = prediction_df.merge(out_df, on="id", how="left")

    print(result.head())

    temp = result.loc[:, "predicted_y"].isnull()

    print("null y:", np.sum(temp))

    for idx in range(len(result)):
        if temp[idx] is False:
            result.loc[idx, "predicted_x"] = int(result.loc[idx, "predicted_y"])

    print(result.head(2))

    result = result.drop("predicted_y", axis=1)

    result.columns = ["id", "predicted"]

    print(result.head(4))

    result.to_csv("../data/model_v2_6j_predictions.csv", index=False)
