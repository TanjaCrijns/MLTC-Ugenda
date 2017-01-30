# -*- coding: utf-8 -*-
import MySQLdb
import sklearn
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from BeautifulSoup import BeautifulSoup
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import pickle
from sklearn.ensemble import RandomForestClassifier
import os.path
import datetime
import glob, os
import pandas as pd
from tqdm import tqdm
import random

# --------------------------------------------------  Global declarations  -------------------------------------------------- #
label_names = ["Dans", "Woord", "Theater", "Film", "Muziek", "Party", "Beeldend", "Kids", "Evenement", "Varia"]
db = None

# --------------------------------------------------     Help functions    -------------------------------------------------- #

# Funcion to connect with database
def ConnectDB():
    global db
    db = MySQLdb.connect(host="localhost",
                         user="root",
                         db="ugenda",
                         charset = "utf8" )
    return db.cursor()

# Function that deletes duplicates from dataset. Duplicates with a different category are joined (Title and introtext)
def combine_labels(data_rows, remove_dup):
    # Put id's in list
    ids = zip(*data_rows)[0]

    # Make two dictionaries for labels and the data with id as keys\
    id_cat_dict = {id:[] for id in ids}
    id_data_dict = {id:[] for id in ids}


    for row in data_rows:
        event_id = row[0]
        category = row[6]

        #put category in dictionary
        id_cat_dict[event_id].append(category)
        #fix for listfailure python
        #id_cat_dict[event_id] = list(set(id_cat_dict[event_id]))
        #id_data_dict[event_id] = row[0:-1]

        # Filter html with beautifulsoup, filter tag $nbsp manually, saw the tag often but not removed by Beautifuloup
        id_data_dict[event_id] = (row[0], BeautifulSoup(row[1]).getText().replace("&nbsp", ""), row[2], row[3], row[4], BeautifulSoup(row[5]).getText().replace("&nbsp", ""), row[6], row[7])

    events = []
    labels = []
    for id in sorted(id_cat_dict.keys()):
        events.append(id_data_dict[id])
        labels.append(id_cat_dict[id])

    if remove_dup:
        without_dup_text_title = []

        without_dup_events = []
        without_dup_labels = []
        for index, event in enumerate(events):
            if not event[1] + event[5] in without_dup_text_title:
                without_dup_text_title.append(event[1] + event[5])
                without_dup_events.append(event)
                without_dup_labels.append(labels[index])
        return without_dup_events, without_dup_labels
    else:
        return events, labels

# Function that puts the content of a textfile into a list
def read_list_text(file):
    with open(file, 'r') as f:
        plain_list = f.readlines()
    return plain_list

# Function that shows how many labels there are in each category
def get_numbers_categories(labels) :
    dans = 0
    woord = 0
    theater = 0
    film = 0
    muziek=0
    party=0
    beeldend=0
    kids=0
    evenement=0
    varia =0
    for label in labels:
        if 0 in label:
            dans = dans+1
        if 1 in label:
            woord = woord+1
        if 2 in label:
            theater = theater+1
        if 3 in label:
            film = film+1
        if 4 in label:
            muziek = muziek+1
        if 5 in label:
            party = party+1
        if 6 in label:
            beeldend = beeldend+1
        if 7 in label:
            kids = kids+1
        if 8 in label:
            evenement = evenement+1
        if 9 in label:
            varia = varia+1

    print "dans =" + str(dans) + "\nwoord =" + str(woord) + "\ntheater =" + str(theater) + "\nfilm =" + str(film) + "\nmuziek =" + str(muziek) + "\nparty =" + str(party) + "\nparty =" + str(party) + "\nbeeldend =" + str(beeldend) + "\nkids =" + str(kids) + "\nevenement =" + str(evenement) + "\nvaria =" + str(varia)
    totaal = dans + woord + theater + film + muziek + party + beeldend + kids+ evenement + varia
    print "Totaal aantal labels:" + str(totaal)

# Function that receives the date of an event and return "weekend", "weekdag" or "onbekendedatum"
def date_one_event(date):
    if date:
        day = date.weekday()
        if day > 3:
            return "weekend"
        else:
            return "weekdag"
    else:
        return "onbekendedatum"

# Function that receives the time of an event and returns "ochtendtijd", "middagtijd", "avonddtijd" or "onbekendetijd"
def time_one_event(time):
    if time is not None:
        time = str(time)
        time = time.split(":")
        hour = int(time[0])
        if hour < 12:
            return "ochtendtijd"
        if 12 < hour and hour < 18:
            return "middagtijd"
        else:
            return "avonddtijd"
    else:
        return "onbekendetijd"

# Function that receives the location of an event and returns a generalized version of that location
def location_one_event(location_id, location_list, only_loc):
    path = "D:/Users/Tanja/Documents/Scriptie/mltc-Ugenda/data/barren.txt"
    with open(path, 'r') as f:
        bar_nijmegen = f.readlines()
    if location_id == 0 or location_id == 1:
        return (True,"anderelocatie")
    location_names = [loc[1] for loc in location_list]
    index = [loc[0] for loc in location_list].index(location_id)
    if only_loc == True:
        return (False,location_names[index])
    if "school" in location_names[index] or "college" in location_names[index] or "universiteit" in location_names[index] or "campus" in location_names[index] or "gymnasium" in location_names[index]:
        return (True,"schoollocatie")
    if (any(bar in location_names[index] for bar in bar_nijmegen) or"café".decode('utf-8') in location_names[index] or "cafe" in location_names[index] or " bar " in location_names[index]) and not "CultuurCafé".decode('utf-8') in location_names[index] :
        return (True,"cafelocatie")
    if "kerk" in location_names[index] or "church" in location_names[index] or "kapel" in location_names[index]:
        return (True,"kerklocatie")
    if "bibliotheek" in location_names[index] or "library" in location_names[index]:
        return (True,"boeklocatie")
    if "park" in location_names[index] or "plein" in location_names[index] or "kade" in location_names[index] or "tuin" in location_names[index] or "buiten" in location_names[index] or "berendonck" in location_names[index]:
        return (True,"buitenlocatie")
    if "cinema" in location_names[index] or "bioscoop" in location_names[index] or "filmhuis" in location_names[index]:
        return (True,"bioscooplocatie")
    if "theater" in location_names[index] or "schouwburg" in location_names[index] or "toneel" in location_names[index]:
        return (True,"theaterlocatie")
    if "fabriek" in location_names[index] or "honig" in location_names[index] or "vasim" in location_names[index]:
        return (True,"fabrieklocatie")
    else:
        return (False, location_names[index])



# php failed me here, so extra function for putting the location id in the event list
def add_venues(data, venues):
    data = list(data)
    for index, row in enumerate (data):
        for loc in venues:
            if row[7] == loc[0]:
                listdata = list(data[index])
                listdata[7] = loc[1]
                listdata = tuple(listdata)
                data[index] = listdata
                break

    return tuple(data)

# ----------------------------------------------- Natural language processing ----------------------------------------------- #

# Funtion that filters the dataframe based on certain part of speech (PoS) tags
def filter_dataframe(df):
    #'LET','TW','LID','VG', 'VZ','SPEC(symb)','BW','VNW'
    FILTER = tuple(['LID','VZ', 'VG','BW'])
    df = df[~df['PoS'].str.startswith(FILTER)]
    return df

# Function that takes the frog output and turns it into a plain text string
def frogtoplain() :
    frog_events = []
    os.chdir("../data/frog_output")
    for file in glob.glob("*.out"):
        frog_events.append(file)

    plain_list = []
    path = "D:/Users/Tanja/Documents/Scriptie/mltc-Ugenda/data/frog_output/"
    for file in tqdm(frog_events):
        df = pd.read_csv(path + file, engine='python', sep='\t*', index_col=False, header=None, names=['TokenNumber','Token','Lemma','PoS','PoSConfidence'])
        df = df.drop(['PoSConfidence','TokenNumber'],axis=1)
        df = filter_dataframe(df)

        toString = " ".join( [str(val) for val in list(df['Token'])])
        lowercased = toString.lower()
        plain_list.append(lowercased)

    with open("D:/Users/Tanja/Documents/Scriptie/mltc-Ugenda/data/frogplain.txt", 'w') as f:
        for event in plain_list:
            f.write(event + "\n")


    return plain_list


# ------------------------------------------------------     Main     ------------------------------------------------------ #

if __name__ == "__main__":

    # Only load data from database once, remove event.pkl if you want to load again, make sure to run frog as well
    if os.path.isfile("../data/event.pkl"):
        events, labels = joblib.load("../data/event.pkl")
        print "Data succesfully loaded"
    else:
        cursor = ConnectDB()
        cursor.execute('SELECT a.id, a.title, a.dates, a.times, a.endtimes, a.introtext, c.catid-2, a.locid FROM si3ow_jem_events AS a INNER JOIN si3ow_jem_cats_event_relations AS c ON a.id = c.itemid WHERE catid > 1 AND recurrence_first_id = 0 AND introtext != "" ')
        data_rows = cursor.fetchall()
        data_rows = [event for event in data_rows if len(event[5]) > 20]
        events, labels = combine_labels(data_rows, True)
        joblib.dump((events, labels), "../data/event.pkl")
        print "Data succesfully loaded"
        path = 'D:\Users\Tanja\Documents\Scriptie\mltc-Ugenda\data\events'
        for event in events:
            filename = os.path.join(path, str(event[0]) + ".txt")
            text_file = open(filename, 'w')
            text_file.write(event[1].encode('ascii', 'ignore') + " " + event[5].encode('ascii', 'ignore'))
            text_file.close()
        print "Data succesfully written to textfiles"
    cursor = ConnectDB()
    cursor.execute('SELECT a.id, a.venue FROM si3ow_jem_venues AS a')
    locations = cursor.fetchall()

    print "Number of instances:",len(events)
    get_numbers_categories(labels)
    eenlabel = 0
    tweelabel = 0
    drieofgroterlabel = 0
    for label in labels:
        if len(label) == 1:
            eenlabel = eenlabel +1
        if len(label) == 2:
            tweelabel = tweelabel +1
        if len(label) >  2:
            drieofgroterlabel = drieofgroterlabel +1
    print "een = " + str(eenlabel) + "\ntwee = " + str(tweelabel) + "\ndrieofgroter = " + str(drieofgroterlabel)
    filename = "../data/stopwords.txt"
    with open(filename, 'r') as f:
        stopwords = f.readlines()
    stopwords = [word[:-1] for word in stopwords]

    # If the location is transformed into a feature, it is added twice to the plaintext. If is has remained the original location, it is only added once.
    date_time_list = [date_one_event(event[2])+ " " + time_one_event(event[3]) + " " for event in events]
    date_time_loc_list = []
    for event in events:
        if location_one_event(event[7],locations, False)[0] == True:
            date_time_loc_list.append(date_one_event(event[2])+ " " + time_one_event(event[3]) + " " + location_one_event(event[7],locations, False)[1] + " ")
        else:
            date_time_loc_list.append(date_one_event(event[2])+ " " + time_one_event(event[3]) + " ")

    loc_list = []
    for event in events:
        if location_one_event(event[7],locations, False)[0] == True:
            loc_list.append(location_one_event(event[7],locations, False)[1] + " ")
        else:
            loc_list.append(" ")

    only_location = [location_one_event(event[7],locations, True)[1] for event in events]
    #temp_plaintext = ["".join(event[1] + " " + event[5]) for event in events]
    #temp_plaintext = frogtoplain()
    temp_plaintext = read_list_text("D:/Users/Tanja/Documents/Scriptie/mltc-Ugenda/data/frogplain.txt")

    #plaintext = [event[1] + " " + event[5] for event in events]
    #plaintext = [event.decode('utf-8') for event in temp_plaintext]
    #plaintext = [date_time_list[y] + " " + date_time_list[y] +  event[1] + " " + event[5] for y, event in enumerate(events)]
    #plaintext = [date_time_list[y] + " " + date_time_list[y] + event.decode('utf-8') for y, event in enumerate(temp_plaintext)]
    #plaintext = [loc_list[y] + " " + loc_list[y] +  " " + only_location[y] + " " + only_location[y] + " " + event[1] + " " + event[5] for y, event in enumerate(events)]
    #plaintext = [loc_list[y] + " " + loc_list[y] +  " " + only_location[y] + " " + only_location[y] + " " + event.decode('utf-8') for y, event in enumerate(temp_plaintext)]
    #plaintext = [date_time_loc_list[y] + " " + date_time_loc_list[y] +  " " + only_location[y] + " " + only_location[y] + " " + event[1] + " " + event[5] for y, event in enumerate(events)]
    plaintext = [date_time_loc_list[y] + " " + date_time_loc_list[y] +  " " + only_location[y] + " " + only_location[y] + " " + event.decode('utf-8') for y, event in enumerate(temp_plaintext)]


    #10 fold test split
    #random state 0: 0.779, 1: 0.797, 2: 0.779 , 3: 0.778, 4: 0.798 , 5: 0.756, 69: 0.786 1337: 0,793,
    kf = KFold(len(plaintext), n_folds=10, shuffle=True, random_state=4)
    f1_score_macro_list  = []
    f1_score_weighted_list = []
    recall_list = []
    precision_list = []
    accuracy_list = []
    for train, test in kf:
        print "Vectorizing"
        v = CountVectorizer(stop_words=stopwords)
        #v = TfidfVectorizer(stop_words=stopwords)
        mb = MultiLabelBinarizer()
        train_text = [plaintext[i] for i in train]
        test_text = [plaintext[i] for i in test]
        binary_labels = mb.fit_transform(labels)
        X_train  = v.fit_transform(train_text)
        X_test = v.transform(test_text)
        y_train = binary_labels[train]
        y_test = binary_labels[test]
        # SVM
        #clf = OneVsRestClassifier(SGDClassifier(random_state=0), n_jobs=-2)
        # Naive bayes
        #clf = OneVsRestClassifier(MultinomialNB(), n_jobs=-2)
        # Logistic regression
        clf = OneVsRestClassifier(LogisticRegression(class_weight=None , random_state=0, solver='liblinear'),n_jobs=-2)
        # Random forest
        #clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=30, random_state=0, class_weight="balanced"), n_jobs=-2)
        print "Fitting"
        clf.fit(X_train, y_train)

        # force_label only with naive bayes and logistic regression because they work with probabilities
        force_label = True
        if force_label:
            probabilities = clf.predict_proba(X_test)
            predictions = np.where(probabilities >= 0.5, 1, 0)
            for (probability, prediction) in zip(probabilities, predictions):
                if np.sum(prediction) == 0:
                    index = np.argmax(probability)
                    prediction[index] = 1
                    #print "Label added", probability[index]
        else:
            predictions = clf.predict(X_test)


        print metrics.classification_report(y_test, predictions, target_names=label_names)
        accuracy = metrics.accuracy_score(y_test, predictions, normalize=True, sample_weight=None)
        accuracy_list.append(accuracy)
        f1_score_weighted = metrics.f1_score(y_test, predictions, average='weighted', sample_weight=None)
        f1_score_weighted_list.append(f1_score_weighted)
        recall = metrics.recall_score(y_test, predictions, pos_label=1, average='weighted', sample_weight=None)
        recall_list.append(recall)
        precision = sklearn.metrics.precision_score(y_test, predictions, pos_label=1, average='weighted', sample_weight=None)
        precision_list.append(precision)

    overall_accuracy = np.mean(accuracy_list)
    overall_f1_weighted = np.mean(f1_score_weighted)
    overall_recall = np.mean(recall_list)
    overall_precision = np.mean(precision_list)

    print "Accuracy=", overall_accuracy, "\n", "Weighted f1=", overall_f1_weighted, "\n", "Recall=", overall_recall, "\n", "Precision=", overall_precision