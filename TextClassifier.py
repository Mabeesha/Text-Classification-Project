import os
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
#import cPickle as c

# Creates the attribute set
def create_attributes():
    # taking all the words to an array
    file_name = "trainset.txt"
    f = open(file_name)
    temp_array = f.read().split("\t")
    words = []
    for string in temp_array:
        temp_words = string.split(" ")
        for i in range(len(temp_words)):
            temp_words[i] = temp_words[i].lower()
        words += temp_words

    # removing non-alphabatic elements from the array
    for i in range(len(words)):
        if not words[i].isalpha():
            words[i] = ""

    attributes = Counter(words) #returns a dictionary
    del attributes[""]
    attributes = attributes.most_common(3000)
    return attributes

# creates the data set from the given file
def create_dataset(attributes):
    feature_set = []
    labels = []

    file_name = "trainset.txt"
    f = open(file_name)
    for line in f:
        words = []
        temp_array = line.split("\t")
        for string in temp_array:
            temp_words = string.split(" ")
            for i in range(len(temp_words)):
                temp_words[i] = temp_words[i].lower()
            words += temp_words
        # creating the labels array

        if temp_array[0] == "-1":
            labels.append(0)
        elif temp_array[0] == "+1":
            labels.append(1)
        else:
            labels.append(2)

        # creating the feature set
        data = []
        for entry in attributes:
            data.append(words.count(entry[0]))
        feature_set.append(data)

    # print(labels)
    # print(feature_set[0])
    return feature_set, labels

# Creates the input for the predictions
def create_Prediction_Input(attributes):
    feature_set = []

    file_name = "testsetwithoutlabels.txt"
    f = open(file_name)
    for line in f:
        words = []
        temp_array = line.split("\t")
        for string in temp_array:
            temp_words = string.split(" ")
            for i in range(len(temp_words)):
               temp_words[i] = temp_words[i].lower()
            words += temp_words

        # creating the input feature set
        data = []
        for entry in attributes:
            data.append(words.count(entry[0]))
        feature_set.append(data)

    return feature_set

def print_predictions(preds):
    f = open("e14390.txt", "w+")
    for i in range(len(preds)):
        if preds[i] == 0:
            f.write("-1\n")
        elif preds[i] == 1:
            f.write("+1\n")
        else:
            f.write("0\n")
    f.close()


# create_dataset(create_attributes())
attributes = create_attributes()
features, labels = create_dataset(attributes)

# splitting training set and test set
x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.25)

# Trying different models - naive bayes, NN, LogisticRegression
clf = MultinomialNB() # 0.9
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3000, 2), random_state=1) #0.875 - 2 layers
# clf = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial') # 0.85

clf.fit(x_train, y_train)

# Getting a score
preds = clf.predict(x_test)
print(accuracy_score(y_test, preds))
print(preds)

# gives predictions for unlabeles data set as a text file
Prediction_set = create_Prediction_Input(attributes)
preds2 = clf.predict(Prediction_set)
print_predictions(preds2)
