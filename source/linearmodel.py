from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
from six.moves.urllib.request import urlopen
import json
import random
import numpy as np
import tensorflow as tf
import collections
from pandas.io.json import json_normalize

TRAINING_SIZE=10

# Data sets
PARTICIPANT_TRAINING = "PARTICIPANT_TRAINING.csv"
PARTICIPANT_TRAINING_URL = "https://v3v10.vitechinc.com/solr/v_participant/select?indent=on&wt=json" + \
    "&q=state:Alaska" + "&rows=50" + "&fl=name,DOB,sex,latitude,longitude"

PARTICIPANT_TEST = "PARTICIPANT_TEST.csv"
PARTICIPANT_TEST_URL = PARTICIPANT_TRAINING_URL

preconditions=['R19.7', 'E11.65', 'F10.121', 'R00.8', 'F14.121', 'T85.622', 'T84.011', 'R04.2', 'B20.1', 'G30.0', 'N18.9', 'G80.4', 'B18.1', 'G47.33', 'M05.10', 'Z91.010', 'S62.308', 'R00.0']
keys=["latitude","longitude","PURCHASED","EMPLOYMENT_STATUS", "ANNUAL_INCOME","HEIGHT","WEIGHT", "PEOPLE_COVERED", "OPTIONAL_INSURED", "DOB", "BRONZE", "SILVER", "GOLD", "PLATINUM"]+preconditions
#keys = ["sex", "EMPLOYMENT_STATUS", "TOBACCO", "MARITAL_STATUS", "latitude", "longitude"]
plan_ranks = ["BRONZE", "SILVER", "GOLD", "PLATINUM"]
heading = [0, (len(keys))] + plan_ranks
plan_to_nums_dict_normal_case = {"Bronze": 0,
                                 "Silver": 1,
                                 "Gold": 2,
                                 "Platinum": 3}
nums_to_plan = {0: "Bronze",
                1: "Silver",
                2: "Gold",
                3: "Platinum"}
str_to_nums_dict = {"sex": {"M": 0, "F": 1},
                    "EMPLOYMENT_STATUS": {"Unemployed": 0, "Employed": 1},
                    "TOBACCO": {"NO": 0, "YES": 1},
                    "MARITAL_STATUS": {"S": 0, "M": 1},
                    "PURCHASED" : plan_to_nums_dict_normal_case,
                    "Risk_factor" : {"Low": 1, "Medium": 2, "High": 3}}

platinums = []
golds = []
silvers = []
bronzies = []
passed_data = []

group_participants = []
group_data = []

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
    global group_participants
    global group_data
    global platinums
    global golds
    global silvers
    global bronzies
    group_participants = urlopen("https://v3v10.vitechinc.com/solr/v_participant/select?indent=on&wt=json&q=*:*&rows="+str(TRAINING_SIZE)).read()
    group_data = json.loads(group_participants)['response']['docs']
    for participant in group_data:
        detail = urlopen("https://v3v10.vitechinc.com/solr/v_participant_detail/select?indent=on&wt=json&q=id="+str(participant['id'])+"&*:*&rows=1").read()
        detail = flatten_json(json.loads(detail)['response']['docs'][0])
        for key in detail:
            #print (key)
            for precon in preconditions:
                participant[precon]=0
            if key == "PRE_CONDITIONS":
                abc = json.loads(detail[key])
                for precon in abc:
                    participant[precon["ICD_CODE"]]=str_to_nums_dict["Risk_factor"][precon["Risk_factor"]]
                    #print(str_to_nums_dict["Risk_factor"][precon["Risk_factor"]])
            if key in keys:
                participant[key]=detail[key]
        quote = urlopen("https://v3v10.vitechinc.com/solr/v_quotes/select?indent=on&wt=json&q=id="+str(participant['id'])+"&*:*&rows=1").read()
        quote = flatten_json(json.loads(quote)['response']['docs'][0])
        for key in quote:
            if key == "PURCHASED":
                if key == "Platinum":
                    platinums.append(participant)
                elif key == "Gold":
                    golds.append(participant)
                elif key == "Silver":
                    silvers.append(participant)
                else:
                    bronzies.append(participant)
            if key in keys:
                participant[key]=quote[key]

def makeHeader(loopNum, lastLoopPrediction):
    num_vals = 4
    if loopNum > 0:
        num_vals = 10
    header = [len(passed_data)] + [len(keys)]
    if loopNum <= 2:
        for index in range(num_vals):
            header+= [index+lastLoopPrediction + (10 ** loopNum)]
    return header

def makeBuckets(data, which_plan, loopNum, lastLoopPrediction):
    global passed_data
    division = 10 ** (2 - loopNum)
    for participant in data:
        participant_copy = copy.deepcopy(participant)
        if nums_to_plan[which_plan] == "Bronze":
            participant_copy["BRONZE"] = int(participant_copy["BRONZE"]/division)
        elif nums_to_plan[which_plan] == "Silver":
            participant_copy["SILVER"] = int(participant_copy["SILVER"]/division)
        elif nums_to_plan[which_plan] == "Gold":
            participant_copy["GOLD"] = int(participant_copy["GOLD"]/division)
        elif nums_to_plan[which_plan] == "Platinum":
            participant_copy["PLATINUM"] = int(participant_copy["PLATINUM"]/division)
        passed_data.append(participant_copy)
    return passed_data

def setup(data, which_plan, loopNum, lastLoopPrediction):

    global heading

    fileName = PARTICIPANT_TRAINING

    if which_plan == -1:
        collapse()
        data = group_data
    else:
        if nums_to_plan[which_plan] == "Bronze":
            fileName = "BRONZIES_" + PARTICIPANT_TRAINING
        elif nums_to_plan[which_plan] == "Silver":
            fileName = "SILVERS_" + PARTICIPANT_TRAINING
        elif nums_to_plan[which_plan] == "Platinum":
            fileName = "GOLDS_" + PARTICIPANT_TRAINING
        else:
            fileName = "PLATINUMS_" + PARTICIPANT_TRAINING
        makeBuckets(data, which_plan, loopNum, lastLoopPrediction)
        heading = makeHeader(loopNum, lastLoopPrediction)
        print(heading)
        data = passed_data

    if which_plan == -1 or not os.path.exists(fileName):
        heading[0] = len(data)
        with open(fileName, "wb") as f:
            for h in heading:
                f.write(b"" + str(h).encode() + b",")
            f.write(b'\n')
            for d in data:
                plan = ""
                for key in d:
                    if key in keys:
                        if key == "DOB":
                            d[key]=d[key][:4]
                        if key in str_to_nums_dict:
                            if key == "PURCHASED":
                                plan = str_to_nums_dict[key]
                            f.write(b"" + str( str_to_nums_dict[key][d[key]] ).encode() + b",")
                        else:
                            f.write(b"" + str(d[key]).encode() + b",")
                if which_plan == -1:
                    f.write(b"" + str(plan[d["PURCHASED"]]).encode() + b"\n")
                else:
                    if nums_to_plan[which_plan] == "Bronze":
                        f.write(b"" + str(d["BRONZE"]).encode() + b"\n")
                    elif nums_to_plan[which_plan] == "Silver":
                        f.write(b"" + str(d["SILVER"]).encode() + b"\n")
                    elif nums_to_plan[which_plan] == "Gold":
                        f.write(b"" + str(d["GOLD"]).encode() + b"\n")
                    else:
                        f.write(b"" + str(d["PLATINUM"]).encode() + b"\n")

    fileName = PARTICIPANT_TEST
    if which_plan != -1:
        if nums_to_plan[which_plan] == "Bronze":
            fileName = "BRONZIES_" + PARTICIPANT_TRAINING
        elif nums_to_plan[which_plan] == "Silver":
            fileName = "SILVERS_" + PARTICIPANT_TRAINING
        elif nums_to_plan[which_plan] == "Platinum":
            fileName = "GOLDS_" + PARTICIPANT_TRAINING
        else:
            fileName = "PLATINUMS_" + PARTICIPANT_TRAINING

    if which_plan == -1 or not os.path.exists(fileName):
        with open(PARTICIPANT_TEST, "wb") as f:
            for h in heading:
                f.write(b"" + str(h).encode() + b",")
            f.write(b'\n')
            for d in data:
                plan = ""
                for key in d:
                    if key in keys:
                        if key == "DOB":
                            d[key]=d[key][:4]
                        if key in str_to_nums_dict:
                            if key == "PURCHASED":
                                plan = str_to_nums_dict[key]
                            f.write(b"" + str( str_to_nums_dict[key][d[key]] ).encode() + b",")
                        else:
                            f.write(b"" + str(d[key]).encode() + b",")
                if which_plan == -1:
                    f.write(b"" + str(plan[d["PURCHASED"]]).encode() + b"\n")
                else:
                    if nums_to_plan[which_plan] == "Bronze":
                        f.write(b"" + str(d["BRONZE"]).encode() + b"\n")
                    elif nums_to_plan[which_plan] == "Silver":
                        f.write(b"" + str(d["SILVER"]).encode() + b"\n")
                    elif nums_to_plan[which_plan] == "Gold":
                        f.write(b"" + str(d["GOLD"]).encode() + b"\n")
                    else:
                        f.write(b"" + str(d["PLATINUM"]).encode() + b"\n")

    # Load datasets.
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=fileName,
        target_dtype=np.int,
        features_dtype=np.float32)

    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=fileName,
        target_dtype=np.int,
        features_dtype=np.float32)

    # Specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column("x", shape=[len(keys)])]

    num_classes = len(plan_ranks)
    model_dir_path = "/tmp/insurance_plan_model"
    if which_plan != -1:
        if nums_to_plan[which_plan] == "Bronze":
            model_dir_path = "/tmp/insurance_plan_model/" + str(loopNum) + "/bronzies"
        elif nums_to_plan[which_plan] == "Silver":
            model_dir_path = "/tmp/insurance_plan_model/" + str(loopNum) + "/silvers"
        elif nums_to_plan[which_plan] == "Gold":
            model_dir_path = "/tmp/insurance_plan_model/" + str(loopNum) + "/golds"
        else:
            model_dir = "/tmp/insurance_plan_model/" + str(loopNum) + "/platinums"
        if loopNum == 0:
            num_classes = 4
        else:
            num_classes = 10

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=num_classes,
                                            model_dir=model_dir_path)
    # Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True)

    # Train model.
    classifier.train(input_fn=train_input_fn, steps=5000)

    # Define the test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(test_set.data)},
        y=np.array(test_set.target),
        num_epochs=1,
        shuffle=False)

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
    
    return accuracy_score

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

def grabPrediction():
    prediction = ""

def writeAllData():
    return group_data

def grabUserData():
    global passed_data
    plan_prices = {}

    # loop over bronzies
    lastPrediction = 0
    for i in range(3):
        print("i = " + str(i))
        print(lastPrediction)
        lastPrediction = int(setup(bronzies, 0, i, lastPrediction))
        passed_data = []
    plan_prices["BRONZE"] = lastPrediction

    print(plan_prices["BRONZE"])

    # loop over silvers
    lastPrediction = 0
    for i in range(3):
        lastPrediction = int(setup(silvers, 1, i, lastPrediction))
        passed_data = []
    plan_prices["SILVER"] = lastPrediction

    
    print(plan_prices["SILVER"])

    # loop over golds
    lastPrediction = 0
    for i in range(3):
        lastPrediction = int(setup(golds, 2, i, lastPrediction))
        passed_data = []
    plan_prices["GOLD"] = lastPrediction

    
    print(plan_prices["GOLD"])

    # loop over plats
    lastPrediction = 0
    for i in range(3):
        lastPrediction = int(setup(platinums, 3, i, lastPrediction))
        passed_data = []
    plan_prices["PLATINUMS"] = lastPrediction

    
    print(plan_prices["PLATINUM"])

    return plan_prices


if __name__ == "__main__":
    #collapse()
    setup(group_data, -1, -1, -1)
    grabUserData()