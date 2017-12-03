from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
from six.moves.urllib.request import urlopen
import json
import random
import numpy as np
import tensorflow as tf
import collections
from pandas.io.json import json_normalize

TRAINING_SIZE=1

# Data sets
PARTICIPANT_TRAINING = "PARTICIPANT_TRAINING.csv"
PARTICIPANT_TRAINING_URL = "https://v3v10.vitechinc.com/solr/v_participant/select?indent=on&wt=json" + \
    "&q=state:Alaska" + "&rows=50" + "&fl=name,DOB,sex,latitude,longitude"

PARTICIPANT_TEST = "PARTICIPANT_TEST.csv"
PARTICIPANT_TEST_URL = PARTICIPANT_TRAINING_URL

keys=["latitude","longitude","PURCHASED","EMPLOYMENT_STATUS"]
#keys = ["sex", "EMPLOYMENT_STATUS", "TOBACCO", "MARITAL_STATUS", "latitude", "longitude"]
plan_ranks = ["BRONZE", "SILVER", "GOLD", "PLATINUM"]
heading = [0, (len(keys))] + plan_ranks
str_to_nums_dict = {"sex": {"M": 0, "F": 1},
                    "EMPLOYMENT_STATUS": {"Unemployed": 0, "Employed": 1},
                    "TOBACCO": {"NO": 0, "YES": 1},
                    "MARITAL_STATUS": {"S": 0, "M": 1} }
plan_to_nums_dict = {"BRONZE": 0,
                     "SILVER": 1,
                     "GOLD": 2,
                     "PLATINUM": 3 }

group_data=[]

#credit to Amir Ziai
def flatten_json(y):
    out = {}
    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x
    flatten(y)
    return out

def collapse():
    group_participants = urlopen("https://v3v10.vitechinc.com/solr/v_participant/select?indent=on&wt=json&q=*:*&rows="+str(TRAINING_SIZE)+"&fl=*").read()
    #details_participants = urlopen()
    group_data = json.loads(group_participants)['response']['docs']
    for participant in group_data:
        detail = urlopen("https://v3v10.vitechinc.com/solr/v_participant_detail/select?indent=on&wt=json&q=id="+str(participant['id'])+"&*:*&rows=100&fl=*").read()
        detail = flatten_json(json.loads(detail)['response']['docs'][0])
        for key in detail:
            if key in keys:
                participant[key]=detail[key]
        quote = urlopen("https://v3v10.vitechinc.com/solr/v_quotes/select?indent=on&wt=json&q=id="+str(participant['id'])+"&*:*&rows=100&fl=*").read()
        quote = flatten_json(json.loads(quote)['response']['docs'][0])
        for key in quote:
            if key in keys:
                participant[key]=quote[key]

def main():
    # If the training and test sets aren't stored locally, download them.
    if not os.path.exists(PARTICIPANT_TRAINING):
        raw = urlopen(PARTICIPANT_TRAINING_URL).read()
        raw = json.loads(raw.decode('utf-8'))['response']['docs']
        raw = [{key: i[key] for key in keys} for i in raw]
        heading[0] = len(raw)
        with open(PARTICIPANT_TRAINING, "wb") as f:
            for h in heading:
                f.write(b"" + str(h).encode() + b",")
            f.write(b'\n')
            for d in raw:
                for key in d:
                    if key in keys:
                        if key in str_to_nums_dict:
                            f.write(b"" + str( str_to_nums_dict[key][d[key]] ).encode() + b",")
                        else:
                            f.write(b"" + str(d[key]) + b",")
                f.write(str(random.randint(0,3)).encode() + b'\n')
    if not os.path.exists(PARTICIPANT_TEST):
        raw = urlopen(PARTICIPANT_TRAINING_URL).read()
        raw = json.loads(raw.decode('utf-8'))['response']['docs']
        raw = [{key: i[key] for key in keys} for i in raw]
        heading[0] = len(raw)
        with open(PARTICIPANT_TEST, "wb") as f:
            for h in heading:
                f.write(b"" + str(h).encode() + b",")
            f.write(b'\n')
            for d in raw:
                for key in d:
                    if key in keys:
                        if key in str_to_nums_dict:
                            f.write(b"" + str( str_to_nums_dict[key][d[key]] ).encode() + b",")
                        else:
                            f.write(b"" + str(d[key]) + b",")
                f.write(str(random.randint(0,3)) + b'\n')

    # Load datasets.
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=PARTICIPANT_TRAINING,
        target_dtype=np.int,
        features_dtype=np.float32)

    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=PARTICIPANT_TEST,
        target_dtype=np.int,
        features_dtype=np.float32)

    # Specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column("x", shape=[len(keys)])]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=len(plan_ranks),
                                            model_dir="/tmp/insurance_plan_model")
    # Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True)

    # Train model.
    classifier.train(input_fn=train_input_fn, steps=2000)

    # Define the test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(test_set.data)},
        y=np.array(test_set.target),
        num_epochs=1,
        shuffle=False)

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

    # Classify two new flower samples.
    # new_samples = np.array(
    #     [[6.4, 3.2, 4.5, 1.5],
    #      [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
    # predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": new_samples},
    #     num_epochs=1,
    #     shuffle=False)

    # predictions = list(classifier.predict(input_fn=predict_input_fn))
    # predicted_classes = [p["classes"] for p in predictions]

    # print(
    #     "New Samples, Class Predictions:    {}\n"
    #     .format(predicted_classes))

    # print()


if __name__ == "__main__":
    collapse()
    #main()