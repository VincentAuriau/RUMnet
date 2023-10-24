#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import joblib
import pickle
import tensorflow as tf
import json
import pandas as pd
import numpy as np
import csv
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import count_params

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

from matplotlib import pyplot as plt
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# from .synthetic.data_preparation import prediction, compute_loss
from .models import create_model_enh, create_model


def scce_with_ls(y, y_hat):
    y = tf.one_hot(tf.cast(y, tf.int32), 3)
    return tf.keras.losses.CategoricalCrossentropy(
        y, y_hat, from_logits=False, label_smoothing=0.03
    )


def split_data_X(X, splitNumber, paramsGeneral, paramsExperiment, RF_indicator):
    sizeObservation = paramsExperiment["assortmentSettings"]["number_samples"]
    sizeTest = int(paramsGeneral["crossVal"]["fractionTest"] * sizeObservation)
    sizeVal = int(paramsGeneral["crossVal"]["fractionVal"] * sizeObservation)
    sizeTrain = sizeObservation - sizeTest - sizeVal

    if RF_indicator == 1:
        Xval = X.take(
            range(splitNumber * sizeVal, (splitNumber + 1) * sizeVal), mode="wrap", axis=0
        )
        Xtrain = X.take(
            range((splitNumber + 1) * sizeVal, (splitNumber + 1) * sizeVal + sizeTrain),
            mode="wrap",
            axis=0,
        )
        Xtest = X.take(
            range(
                (splitNumber + 1) * sizeVal + sizeTrain,
                (splitNumber + 1) * sizeVal + sizeTrain + sizeTest,
            ),
            mode="wrap",
            axis=0,
        )
    else:
        Xval = [
            X[j].take(
                range(splitNumber * sizeVal, (splitNumber + 1) * sizeVal), mode="wrap", axis=0
            )
            for j in range(len(X))
        ]
        Xtrain = [
            X[j].take(
                range((splitNumber + 1) * sizeVal, (splitNumber + 1) * sizeVal + sizeTrain),
                mode="wrap",
                axis=0,
            )
            for j in range(len(X))
        ]
        Xtest = [
            X[j].take(
                range(
                    (splitNumber + 1) * sizeVal + sizeTrain,
                    (splitNumber + 1) * sizeVal + sizeTrain + sizeTest,
                ),
                mode="wrap",
                axis=0,
            )
            for j in range(len(X))
        ]

    return Xval, Xtrain, Xtest


def split_data_XY_enh(X, Y, split_nb, test_size, val_size, RF_indicator):
    n_obs = X[0].shape[0]
    test_len = int(test_size * n_obs)
    val_len = int(val_size * n_obs)
    train_len = n_obs - test_len - val_len

    if RF_indicator == 1:
        Xval = X.take(range(split_nb * val_len, (split_nb + 1) * val_len), mode="wrap", axis=0)
        Xtrain = X.take(
            range((split_nb + 1) * val_len, (split_nb + 1) * val_len + train_len),
            mode="wrap",
            axis=0,
        )
        Xtest = X.take(
            range(
                (split_nb + 1) * val_len + train_len,
                (split_nb + 1) * val_len + train_len + test_len,
            ),
            mode="wrap",
            axis=0,
        )
    else:
        Xval = [
            X[j].take(range(split_nb * val_len, (split_nb + 1) * val_len), mode="wrap", axis=0)
            for j in range(len(X))
        ]
        Xtrain = [
            X[j].take(
                range((split_nb + 1) * val_len, (split_nb + 1) * val_len + train_len),
                mode="wrap",
                axis=0,
            )
            for j in range(len(X))
        ]
        Xtest = [
            X[j].take(
                range(
                    (split_nb + 1) * val_len + train_len,
                    (split_nb + 1) * val_len + train_len + test_len,
                ),
                mode="wrap",
                axis=0,
            )
            for j in range(len(X))
        ]

    Yval = Y.take(range(split_nb * val_len, (split_nb + 1) * val_len), mode="wrap", axis=0)
    Ytrain = Y.take(
        range((split_nb + 1) * val_len, (split_nb + 1) * val_len + train_len),
        mode="wrap",
        axis=0,
    )
    Ytest = Y.take(
        range(
            (split_nb + 1) * val_len + train_len, (split_nb + 1) * val_len + train_len + test_len
        ),
        mode="wrap",
        axis=0,
    )

    return (Xval, Xtrain, Xtest), (Yval, Ytrain, Ytest)


def split_data_Y(Y, splitNumber, paramsGeneral, paramsExperiment):
    sizeObservation = paramsExperiment["assortmentSettings"]["number_samples"]
    sizeTest = int(paramsGeneral["crossVal"]["fractionTest"] * sizeObservation)
    sizeVal = int(paramsGeneral["crossVal"]["fractionVal"] * sizeObservation)
    sizeTrain = sizeObservation - sizeTest - sizeVal

    Yval = Y.take(range(splitNumber * sizeVal, (splitNumber + 1) * sizeVal), mode="wrap", axis=0)
    Ytrain = Y.take(
        range((splitNumber + 1) * sizeVal, (splitNumber + 1) * sizeVal + sizeTrain),
        mode="wrap",
        axis=0,
    )
    Ytest = Y.take(
        range(
            (splitNumber + 1) * sizeVal + sizeTrain,
            (splitNumber + 1) * sizeVal + sizeTrain + sizeTest,
        ),
        mode="wrap",
        axis=0,
    )

    return Yval, Ytrain, Ytest


def plot_history(history, iterNumber, paramsArchitecture, paramsExperiment):
    """
    Plot history of training
    """
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.savefig(
        "output/"
        + paramsExperiment["testGroup"]
        + "/"
        + paramsExperiment["testName"]
        + "/"
        + paramsArchitecture["name"]
        + "/loss_"
        + str(iterNumber)
        + ".png"
    )
    plt.clf()


def plot_history_enh(history, save_to=None, save_name=None):
    """
    Plot history of training
    """
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")

    if save_to is not None and save_name is not None:
        plt.savefig(os.path.join(save_to, save_name + ".png"))
    plt.clf()


def model_fit(
    Xtrain, Xval, Ytrain, Yval, iterNumber, paramsGeneral, paramsArchitecture, paramsExperiment
):
    """
    Single run for cross validation
    """
    print("Num Epch in fit", paramsGeneral["training"]["numberEpochs"])

    ##### Create model
    model = create_model(paramsArchitecture, paramsGeneral, paramsExperiment)
    # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, label_smoothing=paramsGeneral["training"]["label_smoothing"]
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=paramsGeneral["training"]["learningRate"]
    )  # # # #tf.keras.optimizers.legacy.Adam(learning_rate=paramsGeneral['training']['learningRate'])  #
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"]
        # metrics=["accuracy",
        #         lambda x,y:mse(x,y,paramsExperiment['assortmentSettings']["assortment_size"])
        #          ]
    )
    es = EarlyStopping(
        monitor="val_loss",
        patience=paramsGeneral["training"]["early_stopping_patience"],
        verbose=0,
        restore_best_weights=True,
    )
    callbacks = []
    if paramsGeneral["training"]["earlyStopping"]:
        callbacks += [es]

    ###### Fitting model
    start = time.time()
    history = model.fit(
        Xtrain,
        Ytrain,
        validation_data=(Xval, Yval),
        batch_size=paramsGeneral["training"]["batchSize"],
        epochs=paramsGeneral["training"]["numberEpochs"],
        callbacks=callbacks,
        verbose=2,  # 2,
        shuffle=True,
    )

    model.save(
        "output/"
        + paramsExperiment["testGroup"]
        + "/"
        + paramsExperiment["testName"]
        + "/"
        + paramsArchitecture["name"]
        + "/model_"
        + str(iterNumber),
        save_format="tf",
    )
    end = time.time()

    ###### Plot history
    plot_history(history, iterNumber, paramsArchitecture, paramsExperiment)

    ###### Save history
    hist_df = pd.DataFrame(history.history)
    hist_csv_file = (
        "output/"
        + paramsExperiment["testGroup"]
        + "/"
        + paramsExperiment["testName"]
        + "/"
        + paramsArchitecture["name"]
        + "/history_"
        + str(iterNumber)
        + ".csv"
    )
    with open(hist_csv_file, mode="w") as f:
        hist_df.to_csv(f)

    ###### Count number of parameters
    if iterNumber == 0:
        with open(
            "output/"
            + paramsExperiment["testGroup"]
            + "/"
            + paramsExperiment["testName"]
            + "/"
            + paramsArchitecture["name"]
            + "/number_parameters.csv",
            "a",
        ) as fd:
            writer = csv.writer(fd)
            count = np.sum([count_params(p) for p in model.trainable_weights])
            writer.writerow(["numberParameters", count])

    return model, end - start, es.stopped_epoch


def model_fit_enh(
    Xtrain,
    Xval,
    Ytrain,
    Yval,
    model_class,
    model_hyperparams,
    n_epochs,
    lr,
    batch_size,
    label_smoothing=0.0,
    early_stopping_patience=10,
    early_stopping=True,
):
    """
    Single run for cross validation
    """
    print(f"Num Epch in fit: {n_epochs}")

    ##### Create model
    model = create_model_enh(model_class=model_class, model_hyperparams=model_hyperparams)
    # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, label_smoothing=label_smoothing
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr
    )  # # # #tf.keras.optimizers.legacy.Adam(learning_rate=paramsGeneral['training']['learningRate'])  #
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"]
        # metrics=["accuracy",
        #         lambda x,y:mse(x,y,paramsExperiment['assortmentSettings']["assortment_size"])
        #          ]
    )
    es = EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        verbose=0,
        restore_best_weights=True,
    )
    callbacks = []
    if early_stopping:
        callbacks += [es]

    ###### Fitting model
    start = time.time()
    history = model.fit(
        Xtrain,
        Ytrain,
        validation_data=(Xval, Yval),
        batch_size=batch_size,
        epochs=n_epochs,
        callbacks=callbacks,
        verbose=2,  # 2,
        shuffle=True,
    )
    end = time.time()

    return model, history, end - start, es.stopped_epoch


def save_plot_training(model, history, save_directory, save_name):
    model.save(
        os.path.join(save_directory, "model_" + save_name),
        save_format="tf",
    )

    ###### Plot history
    plot_history_enh(history, save_to=save_directory, save_name="loss_" + save_name)

    ###### Save history
    hist_df = pd.DataFrame(history.history)
    hist_csv_file = os.path.join(save_directory, "/history_" + save_name + ".csv")
    with open(hist_csv_file, mode="w") as f:
        hist_df.to_csv(f)


def model_fit_tree(
    Xtrain, Xval, Ytrain, Yval, iterNumber, paramsGeneral, paramsArchitecture, paramsExperiment
):
    ##### Create model
    model = RandomForestClassifier(
        n_estimators=paramsArchitecture["param"]["n_estimators"],
        max_depth=paramsArchitecture["param"]["max_depth"],
        verbose=2,
        criterion="entropy",
        random_state=42,
    )

    ##### Fitting model
    start = time.time()
    model.fit(Xtrain, Ytrain)
    end = time.time()
    dirName = (
        "output/"
        + paramsExperiment["testGroup"]
        + "/"
        + paramsExperiment["testName"]
        + "/"
        + paramsArchitecture["name"]
    )
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    # joblib.dump(model,'output/'+paramsExperiment['testGroup']+'/'+paramsExperiment['testName']+'/'+paramsArchitecture['name']+'/model_'+str(iterNumber)+'.joblib')

    return model, end - start


def cross_validate(X, Y, paramsGeneral, paramsArchitecture, paramsExperiment):
    tol = paramsGeneral["training"]["tol"]
    assortment_size = paramsExperiment["assortmentSettings"]["number_products"]
    data_rows = []

    assortment_size += 1
    for i in range(paramsGeneral["crossVal"]["numberFold"]):
        print("Starting cross fold number {}.".format(i + 1))

        Xval, Xtrain, Xtest = split_data_X(
            X, i, paramsGeneral, paramsExperiment, paramsArchitecture["modelName"] == "RandomForest"
        )
        Yval, Ytrain, Ytest = split_data_Y(Y, i, paramsGeneral, paramsExperiment)

        #### One-hot-encoding of Y
        Ytrain_ohe = tf.keras.utils.to_categorical(
            Ytrain, num_classes=paramsExperiment["assortmentSettings"]["assortment_size"]
        )
        Yval_ohe = tf.keras.utils.to_categorical(
            Yval, num_classes=paramsExperiment["assortmentSettings"]["assortment_size"]
        )
        Ytest_ohe = tf.keras.utils.to_categorical(
            Ytest, num_classes=paramsExperiment["assortmentSettings"]["assortment_size"]
        )

        data_cross_val = []

        if paramsArchitecture["modelName"] == "RandomForest":
            Xtrain = np.array(Xtrain)
            Xval = np.array(Xval)
            Xtest = np.array(Xtest)
            Yval = np.array(Yval)
            Ytrain = np.array(Ytrain)
            Ytest = np.array(Ytest)

            #### model fit
            model, time_fit = model_fit_tree(
                Xtrain, Xval, Ytrain, Yval, i, paramsGeneral, paramsArchitecture, paramsExperiment
            )

            #### Evaluate model
            trainAcc = accuracy_score(Ytrain, model.predict(Xtrain))
            valAcc = accuracy_score(Yval, model.predict(Xval))
            testAcc = accuracy_score(Ytest, model.predict(Xtest))

            y_pred = (1 - tol) * model.predict_proba(Xtrain) + tol / paramsExperiment[
                "assortmentSettings"
            ]["number_products"]
            trainLoss = -np.mean(np.log([y_pred[q, Ytrain[q]] for q in range(len(Ytrain))]))
            y_pred = (1 - tol) * model.predict_proba(Xval) + tol / paramsExperiment[
                "assortmentSettings"
            ]["number_products"]
            valLoss = -np.mean(np.log([y_pred[q, Yval[q]] for q in range(len(Yval))]))
            y_pred = (1 - tol) * model.predict_proba(Xtest) + tol / paramsExperiment[
                "assortmentSettings"
            ]["number_products"]
            testLoss = -np.mean(np.log([y_pred[q, Ytest[q]] for q in range(len(Ytest))]))

            stopped_epoch = 0

        else:
            #### model fit
            model, time_fit, stopped_epoch = model_fit_enh(
                Xtrain,
                Xval,
                Ytrain_ohe,
                Yval_ohe,
                i,
                paramsGeneral,
                paramsArchitecture,
                paramsExperiment,
            )
            #### Evaluate model
            trainLoss, trainAcc = model.evaluate(Xtrain, Ytrain_ohe, verbose=0)
            testLoss, testAcc = model.evaluate(Xtest, Ytest_ohe, verbose=0)
            valLoss, valAcc = model.evaluate(Xval, Yval_ohe, verbose=0)

        data_cross_val = [
            {
                "Fold": "Fold " + str(i),
                "train_loss": trainLoss,
                "test_loss": testLoss,
                "val_loss": valLoss,
                "train_acc": trainAcc,
                "test_acc": testAcc,
                "val_acc": valAcc,
                "time_fit": time_fit,
                "number_epochs": stopped_epoch,
            }
        ]

        data_rows += data_cross_val

    data = pd.DataFrame(data_rows)
    data = data.set_index("Fold").transpose()
    temp_mean = data.mean(numeric_only=True, axis=1)
    temp_std = data.std(numeric_only=True, axis=1)
    data["mean"] = temp_mean
    data["std"] = temp_std
    data.to_csv(
        "output/"
        + paramsExperiment["testGroup"]
        + "/"
        + paramsExperiment["testName"]
        + "/"
        + paramsArchitecture["name"]
        + "/results.csv",
        index=True,
    )
    print(
        "file saved at",
        "output/"
        + paramsExperiment["testGroup"]
        + "/"
        + paramsExperiment["testName"]
        + "/"
        + paramsArchitecture["name"]
        + "/results.csv",
    )

    with open(
        "output/"
        + paramsExperiment["testGroup"]
        + "/"
        + paramsExperiment["testName"]
        + "/"
        + paramsArchitecture["name"]
        + "/data_train.pickle",
        "wb",
    ) as handle:
        pickle.dump((Xtrain, Ytrain), handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(
        "output/"
        + paramsExperiment["testGroup"]
        + "/"
        + paramsExperiment["testName"]
        + "/"
        + paramsArchitecture["name"]
        + "/data_test.pickle",
        "wb",
    ) as handle:
        pickle.dump((Xtest, Ytest), handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(
        "output/"
        + paramsExperiment["testGroup"]
        + "/"
        + paramsExperiment["testName"]
        + "/"
        + paramsArchitecture["name"]
        + "/data_val.pickle",
        "wb",
    ) as handle:
        pickle.dump((Xval, Yval), handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Data saved")


def cross_validate_enh(
    X, Y, model_class, model_hyperparams, fit_params, n_folds, test_size, val_size, assortment_size, save_dir
):
    data_rows = []

    assortment_size += 1
    for i in range(n_folds):
        print("Starting cross fold number {}.".format(i + 1))

        (Xval, Xtrain, Xtest), (Yval, Ytrain, Ytest) = split_data_XY_enh(
            X, Y, i, test_size, val_size, str(model_class) == "RandomForest"
        )

        #### One-hot-encoding of Y
        Ytrain_ohe = tf.keras.utils.to_categorical(Ytrain, num_classes=assortment_size)
        Yval_ohe = tf.keras.utils.to_categorical(Yval, num_classes=assortment_size)
        Ytest_ohe = tf.keras.utils.to_categorical(Ytest, num_classes=assortment_size)

        data_cross_val = []

        if str(model_class) == "RandomForest":
            Xtrain = np.array(Xtrain)
            Xval = np.array(Xval)
            Xtest = np.array(Xtest)
            Yval = np.array(Yval)
            Ytrain = np.array(Ytrain)
            Ytest = np.array(Ytest)

            #### model fit
            model, time_fit = model_fit_tree(
                Xtrain, Xval, Ytrain, Yval, i, paramsGeneral, paramsArchitecture, paramsExperiment
            )

            #### Evaluate model
            trainAcc = accuracy_score(Ytrain, model.predict(Xtrain))
            valAcc = accuracy_score(Yval, model.predict(Xval))
            testAcc = accuracy_score(Ytest, model.predict(Xtest))

            y_pred = (1 - tol) * model.predict_proba(Xtrain) + tol / paramsExperiment[
                "assortmentSettings"
            ]["number_products"]
            trainLoss = -np.mean(np.log([y_pred[q, Ytrain[q]] for q in range(len(Ytrain))]))
            y_pred = (1 - tol) * model.predict_proba(Xval) + tol / paramsExperiment[
                "assortmentSettings"
            ]["number_products"]
            valLoss = -np.mean(np.log([y_pred[q, Yval[q]] for q in range(len(Yval))]))
            y_pred = (1 - tol) * model.predict_proba(Xtest) + tol / paramsExperiment[
                "assortmentSettings"
            ]["number_products"]
            testLoss = -np.mean(np.log([y_pred[q, Ytest[q]] for q in range(len(Ytest))]))

            stopped_epoch = 0

        else:
            #### model fit
            model, history, time_fit, stopped_epoch = model_fit_enh(
                Xtrain=Xtrain,
                Xval=Xval,
                Ytrain=Ytrain_ohe,
                Yval=Yval_ohe,
                model_class=model_class,
                model_hyperparams=model_hyperparams**fit_params,
            )

            save_plot_training(model, history, save_dir, f"{i}")
            #### Evaluate model
            trainLoss, trainAcc = model.evaluate(Xtrain, Ytrain_ohe, verbose=0)
            testLoss, testAcc = model.evaluate(Xtest, Ytest_ohe, verbose=0)
            valLoss, valAcc = model.evaluate(Xval, Yval_ohe, verbose=0)

        data_cross_val = [
            {
                "Fold": "Fold " + str(i),
                "train_loss": trainLoss,
                "test_loss": testLoss,
                "val_loss": valLoss,
                "train_acc": trainAcc,
                "test_acc": testAcc,
                "val_acc": valAcc,
                "time_fit": time_fit,
                "number_epochs": stopped_epoch,
            }
        ]

        data_rows += data_cross_val

    data = pd.DataFrame(data_rows)
    data = data.set_index("Fold").transpose()
    temp_mean = data.mean(numeric_only=True, axis=1)
    temp_std = data.std(numeric_only=True, axis=1)
    data["mean"] = temp_mean
    data["std"] = temp_std
    data.to_csv(
        os.path.join(save_dir, "results.csv"),
        index=True,
    )
    print("data saved at", os.path.join(save_dir, "results.csv"))
