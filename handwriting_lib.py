import os
import sys
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# HELPER FUNCTIONS
# ====================================================================================================


def _save_pickled_model(rfc: RandomForestClassifier, accuracy):
    """Helper Function to use the pickle to serialize and save machine model to disk to save time while testing by not having
       to retrain model on every rerun of the program.

       Model Filename: 'finalized_model.sav'
       Accuracy Filename: 'accuracy.txt'
       """
    # save the model to disk
    pickle.dump(rfc, open('finalized_model.sav', 'wb'))
    print("Stored RFC Model inside of 'finalized_model.sav'")

    #save accuracy score to disk as well
    with open('accuracy.txt', 'w') as f:
        f.write(str(accuracy))
    print("Stored RFC Model Accuracy Score inside of 'accuracy.txt'")

def _load_pickled_model():
    """Helper Function to use the pickle to serialize and save machine model to disk to save time while testing by not having
       to retrain model on every rerun of the program.

       Model Filename: 'finalized_model.sav'
       Accuracy Filename: 'accuracy.txt'
       """
    # load the model from disk
    loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
    print("Loaded Serialized Model 'finalized_model.sav'")

    with open('accuracy.txt', 'r') as f:
        accuracy = f.readlines()
    print("Loaded Serialized Model Accuracy 'accuracy.txt'")

    return (loaded_model, float(accuracy[0]))


# ====================================================================================================

# MAIN FUNCTION(S)
# ====================================================================================================


def create_rfc_model(filename: str, print_results: bool = False) -> tuple:
    """
    Primary Function for creating our Random Forest Classifier that we can use for 
    predictions and prototyping.

    Args:
        filename (str): The Path/Name of a CSV file for training/testing our model on.

    Returns:
        tuple: Index 0 is the Random Forest Classifier Model & Index 1 is a float with model accuracy.
    """

    # For the sake of saving compile time while testing and avoiding wasted time training the model every
    # save the model to file and then just load that next time the file is called
    if os.path.isfile('finalized_model.sav'):
        handwriting_model, model_accuracy = _load_pickled_model()
        return (handwriting_model, model_accuracy)
    else:

        if filename[-3:] == "csv":
            mnist_df = pd.DataFrame(pd.read_csv(f"{filename}"))
        else:
            mnist_df = pd.DataFrame(pd.read_csv(f"{filename}.csv"))
        print(mnist_df.shape)

        # feature
        y = mnist_df["label"]

        # labels
        X = mnist_df.drop("label", axis=1)

        # make train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        # Model Creation
        print("Running RandomForestClassifier...")
        rfc = RandomForestClassifier(
            random_state=1, n_estimators=150, criterion="entropy")
        rfc.fit(X_train, y_train)
        print("Model is ready for use")

        # Model Prediction and report
        y_pred = rfc.predict(X_test)
        print(classification_report(y_test, y_pred))

        # Return the rfc model and our accuracy of the rfc model
        forest_accuracy = accuracy_score(y_test, y_pred)
        print(forest_accuracy)

        #Just for debug/additional information purposes
        if print_results == True:
            result = pd.DataFrame(data=zip(y_test, y_pred), columns=[
                                  "Actual", "RFC Prediction"])
            print(result.head())

        _save_pickled_model(rfc, forest_accuracy)
        return (rfc, forest_accuracy)



# ====================================================================================================


if __name__ == "__main__":
    create_rfc_model(filename=sys.argv[1], print_results=True)
