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

label_names = ["Dans", "Woord", "Theater", "Film", "Muziek", "Party", "Beeldend", "Kids", "Evenement", "Varia"]
db = None
def ConnectDB():
    global db
    db = MySQLdb.connect(host="localhost",    # your host, usually localhost
                         user="root",         # your username
                         db="ugenda",         # name of the data base
                         charset = "utf8" )
    return db.cursor()

# Deletes duplicates, duplicates with a different category are joined (Title and introtext)
def combine_labels(data_rows, remove_dup=True):
    # take id's in list
    ids = zip(*data_rows)[0]

    # make two dictionaries with id as keys
    id_cat_dict = {id:[] for id in ids}
    id_data_dict = {id:[] for id in ids}


    for row in data_rows:
        event_id = row[0]
        category = row[6]
        #-1 is de laatste

        #put category in dictionary
        id_cat_dict[event_id].append(category)
        #fix for listfailure python
        #id_cat_dict[event_id] = list(set(id_cat_dict[event_id]))

        #id_data_dict[event_id] = row[0:-1]
        #filter tag $nbsp manually, saw the tag often but not removed by Beautifuloup
        id_data_dict[event_id] = (row[0], BeautifulSoup(row[1]).getText().replace("&nbsp", ""), row[2], row[3], row[4], BeautifulSoup(row[5]).getText().replace("&nbsp", ""), row[6], row[7])
    events = []
    labels = []
    for id in id_cat_dict.keys():
        events.append(id_data_dict[id])
        labels.append(id_cat_dict[id])

    if remove_dup:
        # hulplijst voor vergelijken text en titel
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

def date_one_event(date):
    if date:
        day = date.weekday()
        if day > 3:
            return "weekend"
        else:
            return "weekday"
    else:
        return "unknowndate"

def time_one_event(time):
    if time is not None:
        time = str(time)
        time = time.split(":")
        hour = int(time[0])
        if hour < 12:
            return "morning"
        if 12 < hour and hour < 18:
            return "afternoon"
        else:
            return "evening"
    else:
        return "unknownstartingtime"

#php failed me here, so extra function
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



if __name__ == "__main__":
    #SQL Query, retrieves event id, title, introtext and category id
    if os.path.isfile("../data/event.pkl"):
        events, labels = joblib.load("../data/event.pkl")
        print "Data succesfully loaded"
    else:
        cursor = ConnectDB()
        cursor.execute('SELECT a.id, a.title, a.dates, a.times, a.endtimes, a.introtext, c.catid-2, a.locid FROM si3ow_jem_events AS a INNER JOIN si3ow_jem_cats_event_relations AS c ON a.id = c.itemid WHERE catid > 1 AND recurrence_first_id = 0 AND introtext != "" ')
        data_rows = cursor.fetchall()
        # cursor.execute("SELECT id, venue FROM si3ow_jem_venues")
        # venues = cursor.fetchall()
        # data_rows = add_venues(data_rows, venues)
        # print data_rows[0:10]
        events, labels = combine_labels(data_rows, True)
        joblib.dump((events, labels), "../data/event.pkl")
        print "Data succesfully loaded"

        #do this once with new data
        #path = 'C:\Users\Tanja\Documents\Scriptie\mltc-Ugenda\data\events'
        #for event in events:
        #    filename = os.path.join(path, str(event[0]) + ".txt")
        #    text_file = open(filename, "w")
        #    text_file.write(event[1].encode('ascii', 'ignore') + " " + event[5].encode('ascii', 'ignore'))
        #    text_file.close()
        print "Data succesfully loaded"

    print "Number of instances:",len(events)
    filename = "../data/stopwords.txt"
    with open(filename, 'r') as f:
        stopwords = f.readlines()
    stopwords = [word[:-1] for word in stopwords]
    plaintext = [" ".join(event[1])+ " " + event[5] for event in events]
    #plaintext = [date_one_event(event[2])+ " " + time_one_event(event[3]) for event in events]
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
    print "labels:" + str(len(labels))
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

    print "dans =" + str(dans) + "woord =" + str(woord) + "theater =" + str(theater) + "film =" + str(film) + "muziek =" + str(muziek) + "party =" + str(party) + "party =" + str(party) + "beeldend =" + str(beeldend) + "kids =" + str(kids) + "evenement =" + str(evenement) + "varia =" + str(varia)
    bla = dans + woord + theater + film + muziek + party + beeldend + kids+ evenement + varia
    print "Totaal aantal labels:" + str(bla)

    #10 fold test split
    # meuk = 0
    kf = KFold(len(plaintext), n_folds=6, shuffle=True, random_state=0)
    f1_score_macro_list  = []
    f1_score_weighted_list = []
    recall_list = []
    precision_list = []
    accuracy_list = []
    for train, test in kf:
        # meuk += 1
        # if meuk > 2:
        #     break
        print "Vectorizing"
        #texttitle:
        v = CountVectorizer(stop_words=stopwords)
        #v = TfidfVectorizer(stop_words=stopwords)
        #datetime:
        # v = CountVectorizer()
        #v = TfidfVectorizer()
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
        #clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=20, random_state=0, class_weight="balanced"), n_jobs=-2)
        print "Fitting"
        clf.fit(X_train, y_train)
        #predictions = clf.predict(X_test)
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
        f1_score_macro = metrics.f1_score(y_test, predictions, average='macro', sample_weight=None)
        f1_score_macro_list.append(f1_score_macro)
        f1_score_weighted = metrics.f1_score(y_test, predictions, average='weighted', sample_weight=None)
        f1_score_weighted_list.append(f1_score_weighted)
        recall = metrics.recall_score(y_test, predictions, pos_label=1, average='macro', sample_weight=None)
        recall_list.append(recall)
        precision = sklearn.metrics.precision_score(y_test, predictions, pos_label=1, average='macro', sample_weight=None)
        precision_list.append(precision)

    overall_accuracy = np.mean(accuracy_list)
    overall_f1_macro = np.mean(f1_score_macro)
    overall_f1_weighted = np.mean(f1_score_weighted)
    overall_recall = np.mean(recall_list)
    overall_precision = np.mean(precision_list)

    print "Accuracy=", overall_accuracy, "\n", "Macro f1=", overall_f1_macro, "\n", "Weighted f1=", overall_f1_weighted, "\n", "Recall=", overall_recall, "\n", "Precision=", overall_precision