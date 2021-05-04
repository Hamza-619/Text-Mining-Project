import nltk, re, os, sys, pymongo, json, string, numpy as np, pandas as pd, matplotlib.pyplot as plt

from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from io import StringIO

from sklearn.cross_validation import train_test_split  # splitting dataset
from sklearn.preprocessing import StandardScaler  # feature scaling
from sklearn.linear_model import LogisticRegression  # classifer
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV  # feature selection
from sklearn.metrics import confusion_matrix  # confusion matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from nltk.stem.snowball import SnowballStemmer  # for stemming purpose

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

path = "/home/computer/Desktop/txt_mining_project/final_project_files/Final_Input_Data"
files = [path + "/" + x for x in os.listdir(path) if x.endswith(".html")]

mng_client = pymongo.MongoClient("localhost", 27017)
mng_db = mng_client["document"]
collection_name = "dataset"
db_cm = mng_db[collection_name]
db_cm.remove()

# Read one document and storage preparation function (html to csv)


def doc_prep(url):
    # read file as xml
    html = open(url, encoding="latin-1")
    soup = BeautifulSoup(html, "lxml").get_text()
    #########
    # file format processing

    # eliminate spaces in start & end of the file
    soup = soup.lstrip("\n").rstrip("\n")
    # split file into paragraphs
    para = soup.split(sep="\n\n")
    # eliminate newlines(\n) from paragraphs
    para_tmp = []
    for s in para:
        para_tmp.append(s.replace("\n", " "))
    para = para_tmp
    # remove pages' numbers
    para_tmp = []
    for line in para:
        if len(line.lstrip(" ").rstrip(" ")) > 4:
            para_tmp.append(line)
    para = para_tmp
    #########
    # creating pandas dataframe

    # adding \t\t in the beginning of non-header paragraphs(training purpose)
    para_compressed = [" "]
    for line in para:
        if re.search("^.*[\\t]+.*$", line):
            para_compressed.append(line)
        else:
            para_compressed.append("\t\t" + line)
    # concatenate the paragraphs into one string
    data = "\n".join(para_compressed)
    #########
    # Parsing data using pandas dataframe
    TESTDATA = StringIO(data)
    parser = pd.read_csv(
        TESTDATA, sep="\t", names=["is_header", "section_type", "section"]
    )
    #######dataframe preprocessing
    # treating missing data
    parser["is_header"] = parser["is_header"].fillna(False)
    parser["section_type"] = parser["section_type"].fillna("None")
    parser = parser.dropna()
    # solving type problems(in case)
    parser.is_header = parser.is_header.astype(bool)
    parser.section = parser.section.astype(str)
    parser.section_type = parser.section_type.astype(str)
    return parser

    ######insert documents in database function


def store_content(csv_file):
    mng_client = pymongo.MongoClient("localhost", 27017)
    mng_db = mng_client["document"]
    collection_name = "dataset"
    db_cm = mng_db[collection_name]
    data_json = json.loads(csv_file.to_json(orient="records"))
    # db_cm.remove()
    db_cm.insert(data_json)
    return


######retreive documents from database function


def read_content():
    """ Read from Mongo and Store into DataFrame """
    mng_client = pymongo.MongoClient("localhost", 27017)
    mng_db = mng_client["document"]
    collection_name = "dataset"
    db_cm = mng_db[collection_name]
    cursor = db_cm.find()
    df = pd.DataFrame(list(cursor))
    # Delete the _id
    try:
        del df["_id"]
    except:
        print("This is an error message!")
    return df


for url in files:
    parser = doc_prep(url)
    store_content(parser)
    store_content(parser)
    store_content(parser)
parser = read_content()

####### part_1_processing: ####### train a classifier to decide whether a section is header or not

#######feature preparation

#######define feature functions
def startWithArabic(Instr):
    """
    this function return true if the given string starts with am Arabic numeral
    """
    return Instr[:1].isdigit()


########################################
def startWithRoman(Instr):
    """
    this function return true if the given string starts with Simple Roman numeral followed by period or ' '
    """
    return (
        re.match(
            "^(?=[MDCLXVI])M*(C[MD]|D?C*)(X[CL]|L?X*)(I[XV]|V?I*)[ |.]", Instr[0:3]
        )
        is not None
    )


#######################################
def leadingWhiteSpace(Instr):
    """
    this function return true if the given string starts with WhiteSpace character
    """
    return re.match("^\s", Instr) is not None


########################################
def ellipses(Instr):
    """
    this function return true if the given string contains ellipses
    """
    return re.search(r"(\w+)\.{3,}", Instr) is not None


#######################################
def ContainsComma(Instr):
    return "," in Instr


######################################
def common_words_count(intstr):

    CommonWordsList = stopwords.words("english")
    CourtWordList = [
        "complaint",
        "commission",
        "defendant",
        "appeal",
        "case",
        "action",
        "accused",
        "appellant",
        "crime",
        "answer",
        "brief",
        " claim",
        " collateral",
        " complaint",
        " contract",
        " counsel",
        "count",
        "defendants",
        " evidence",
        "federal question",
        "issue",
        "guilty",
        " judge",
        "conviction",
        "precedent",
        "procedure",
        "jury",
        "sentence",
        "statute",
        "witness",
        "however",
        "federal",
        "law",
        "claim",
        "party",
        "petition",
        "record",
        "statute",
        "judgment",
        "objection",
        "part",
        "party",
        "parties",
        "witness",
        "witnesses",
        "judge",
        "fact",
        "state",
        "sentence",
        "district",
        "trial",
    ]
    CommonWordsList += CourtWordList

    var = intstr
    return sum(1 for word in var.split() if word.lower() in CommonWordsList)


#######################################
def stop_words_count(inputstr):
    """
    this function return the number of stopwords in a String
    """
    return sum(
        1 for word in inputstr.split() if word.lower() in stopwords.words("english")
    )


######################################
def num_Ponctuation(Instr):
    """
    this function return the number of ponctuation characters in a String
    """
    nonpuc = [c for c in Instr if c not in string.punctuation]
    nonpuc = "".join(nonpuc)
    if "." in nonpuc:
        nonpuc = nonpuc + "."  # this line just to ignore the period for one time
    return len(Instr) - len(nonpuc)


#####################################
def special_begin(string):
    """
    this function looks for special begin
    """
    return re.match("^\s*[0-9IVa-f]{1,3}[)|.]\s*.+", string) is not None


#######applying feature functions to dataframe


def feature_application(parser):
    # punctuation count
    parser.loc[:, "num_Ponctuation"] = parser["section"].apply(
        lambda st: num_Ponctuation(st)
    )  # -->this is a bad metric

    # stop_words_count
    parser.loc[:, "common_words_count"] = parser["section"].apply(
        lambda st: common_words_count(st)
    )

    # LeadingAsterisk
    # parser.loc[:,'LeadingAsterisk'] = parser['section'].apply(str.startswith, args='*')

    # # leading arabic numeral
    ArabicNumeral = (
        parser["section"].str.lstrip(" ").apply(lambda st: startWithArabic(st))
    )
    # # leading Roman numeral
    RomanNumerals = (
        parser["section"].str.lstrip(" ").apply(lambda st: startWithRoman(st))
    )
    # leadingNumeral
    parser.loc[:, "LeadingNumeral"] = ArabicNumeral | RomanNumerals

    # endsInPeriod
    # parser.loc[:,'endsInPeriod'] = parser['section'].apply(str.endswith, args='.')

    # leadingWhiteSpace
    # parser['leadingWhiteSpace'] = parser['section'].apply(lambda st: leadingWhiteSpace(st))

    # ellipses
    # parser.loc[:,'ellipses'] = parser['section'].apply(lambda st: ellipses(st))

    # ContainsComma:
    parser.loc[:, "ContainsComma"] = parser["section"].apply(
        lambda st: ContainsComma(st)
    )

    # remove spaces in begening and end of sections to get the real section length
    parser.loc[:, "section"] = parser["section"].map(
        lambda x: x.lstrip(" ").rstrip(" ")
    )
    # section length
    parser.loc[:, "stringLength"] = parser["section"].apply(len)

    # percentCaps (presentation precision = 2)
    parser.loc[:, "percentCaps"] = parser["section"].apply(
        lambda st: round(sum(1 for c in st if c.isupper()) * 100 / len(st), 2)
    )

    # remove remaining page's numbers
    parser = parser.drop(parser[parser["stringLength"] < 4].index)
    # reset index of dataframe
    parser = parser.reset_index(drop=True)

    return parser


parser = feature_application(parser)


###### machine_learning_processing


def non_shuffling_train_test_split(X, y, test_size=0.25):
    i = int((1 - test_size) * X.shape[0]) + 1
    X_train, X_test = np.split(X, [i])
    y_train, y_test = np.split(y, [i])
    return X_train, X_test, y_train, y_test


# read dependant & independants variables
X1 = parser.iloc[:, 3 : len(parser.columns)].values  # [4,5,7,8,9]
Y1 = parser.iloc[:, 0].values
# splitting data set to training set and test set
X1_train, X1_test, Y1_train, Y1_test = non_shuffling_train_test_split(
    X1, Y1, test_size=0.25
)

# feature scaling
sc_X = StandardScaler()
X1_train = sc_X.fit_transform(X1_train)
X1_test = sc_X.transform(X1_test)
# feature selection using recursive feature elimination & training classifer
classifier1 = RFECV(SVC(kernel="linear", random_state=0), scoring="accuracy")
# classifier1 = RFECV(LogisticRegression(random_state=0),scoring='accuracy')
classifier1.fit(X1_train, Y1_train)
# predict the test set result
Y1_pred = classifier1.predict(X1_test)

# to be used in part 2
tested_data, result_part1 = Y1_test, Y1_pred

####### performance of part 1

# confusion Matrix
cm = confusion_matrix(tested_data, result_part1)
print("confusion_matrix:\n", cm)
# accuracy
print("accuracy = ", accuracy_score(tested_data, result_part1))
# recall
print("recall = ", recall_score(tested_data, result_part1))
# precision
print("presicion = ", precision_score(tested_data, result_part1))


#######analysing error cause

diff_true = np.logical_xor(tested_data, result_part1)
# count how many true in diff_true
np.sum(diff_true)
# identifying false results
true_indices = list(np.argwhere(diff_true == True).flatten())
true_indices[:] = [x + len(parser) - len(tested_data) for x in true_indices]
# show false results
parser.loc[true_indices, :]

##-->the cause of this error is no obvious

####### part_2_processing: ####### train a classifier to decide the type of headers
# -->this part will be devided into 2 subparts:
"""
subpart 1 : train the classifier using all the heading section present in the data 
            Goal: analyse performance of classification independantely from part 1
"""
"""
subpart 2 : use the trained classifier with the heading section resulting from part 1 
            Goal: analyse influence of part 1 (error propagation) on the performance of part 2
"""

######subpart 1:

###extracting headers from initial data
Aheader = parser[["section_type", "section"]]
Aheader = Aheader.drop(Aheader[Aheader["section_type"] == "None"].index)

###Features extraction:remove ponctuation -->remove stopwords -->generate stemming -->extract features

# remove ponctuation
Aheader.loc[:, "Non_Ponc"] = Aheader["section"].apply(
    lambda st: re.sub(r"[^a-zA-Z ]", " ", st)
)

# remove stopwords
def clean(st):
    """
    this function returns String without stopwords or numerals
    """
    clean_st = [
        word
        for word in st.split()
        if (word.lower() not in stopwords.words("english")) & (len(word) > 3)
    ]
    return clean_st


Aheader.loc[:, "clean_section"] = Aheader["Non_Ponc"].apply(lambda st: clean(st))

# generate stemming
stemmer = SnowballStemmer("english")
Aheader.loc[:, "stemm"] = Aheader["clean_section"].apply(
    lambda st: [stemmer.stem(word) for word in st]
)

# sorting dataframe
Aheader.sort_values(by="section_type").reset_index(
    drop=True
)  # -->this is not necessary

# extract features(contains steps)
# gathering stemms for each header type in one list
def aggreg(row):
    """
    this function returns a list of concatenated stemms
    """
    flist = []
    for ls in row["stemm"]:
        flist = flist + ls
    return flist


conca = (Aheader.groupby("section_type").apply(lambda lis: aggreg(lis))).to_frame(
    "vocab"
)  # .reset_index()

# sort elements in the vocab listes
conca.loc[:, "vocab"] = conca["vocab"].apply(lambda st: sorted(st))

# eliminate stemms that are not useful for classification(appears just ones)
def useful_words(ls):
    """
    this function takes a list of strings and return a list with strings used more than ones
    """
    bow_transformer = CountVectorizer().fit(ls)
    csr_matrix = bow_transformer.transform([" ".join(ls)])
    tfidf_transfrom = TfidfTransformer().fit_transform(csr_matrix)
    """eliminate"""
    tmp_list = []  # tmp_list contains the elements to be eliminated
    Mc = tfidf_transfrom.tocoo()
    for i in Mc.col:
        if Mc.data[Mc.col == i][0] == Mc.data.min() and Mc.data.min() != Mc.data.max():
            tmp_list.append(bow_transformer.get_feature_names()[i])
    return list(set(ls) - set(tmp_list))


conca.loc[:, "vocab"] = conca["vocab"].apply(lambda st: useful_words(st))

# generate list of features
flist = []
for i in conca.index.values:  # list of categories
    flist = flist + conca.vocab[i]

##generate features/expected-classification-result dataframe
featuresdf = Aheader[["section_type", "section"]]
# insert features in dataframe
for i in flist:
    featuresdf.loc[:, i] = Aheader["section"].apply(
        lambda st: i in " ".join([stemmer.stem(i) for i in st.split(" ")])
    )  # st.lower()
# insert expected results in dataframe
for i in conca.index.values:
    featuresdf.insert(
        0, i, Aheader["section_type"].apply(lambda st: st.lower() == i.lower())
    )
# reset index of dataframe
featuresdf.reset_index(drop=True)

###prediction

# initialise the result dataframe
class_result = featuresdf[["section_type", "section"]].reset_index(drop=True)
class_result = class_result.iloc[int(len(featuresdf) * 0.66) :, :]
class_result.loc[:, "estimated_type"] = "None"

# make estimations
for i in range(0, len(conca)):
    print("treatment of type'", featuresdf.columns[i], "' :done ")
    # splitting data set to training_set and test_set
    X21_train = featuresdf.iloc[: int(len(featuresdf) * 0.66), len(conca) + 2 :].values
    Y21_train = featuresdf.iloc[: int(len(featuresdf) * 0.66), i].values
    ##
    X21_test = featuresdf.iloc[int(len(featuresdf) * 0.66) :, len(conca) + 2 :].values
    Y21_test = featuresdf.iloc[int(len(featuresdf) * 0.66) :, i].values

    # feature scaling
    sc_X = StandardScaler()
    X21_train = sc_X.fit_transform(X21_train)
    X21_test = sc_X.fit_transform(X21_test)
    # feature selection & classifier training
    classifier21 = RFECV(SVC(kernel="linear", random_state=0), scoring="accuracy")
    # classifier21 = RFECV(LogisticRegression(random_state=0),scoring='accuracy')

    classifier21.fit(X21_train, Y21_train)
    # predict the test set result
    Y_pred21 = classifier21.predict(X21_test)

    # store perdiction for class i in the result dataframe
    Y_true = class_result.index[Y_pred21 == True]
    for tr_ind in Y_true:
        if class_result.loc[tr_ind, "estimated_type"] == "None":
            class_result.loc[tr_ind, "estimated_type"] = featuresdf.columns[i]

###performance of subpart 1 (part 2)

# accuracy
print(
    "accuracy = ",
    accuracy_score(class_result.section_type, class_result.estimated_type),
)

######subpart 2:

###extracting headers from first part result
true_indices = list(np.argwhere(result_part1 == True).flatten())
true_indices[:] = [x + len(parser) - len(tested_data) for x in true_indices]
Pheaders = parser.iloc[true_indices, 1:3]
Pheaders = Pheaders.reset_index(drop=True)
# len(Pheaders[Pheaders['section_type']=='None'].index) --> to verify number of wrong result
Pheaders["section_type"].replace("None", "wrongly_estimated", inplace=True)

##generate features/expected-classification-result dataframe
featuresdf = Aheader[["section_type", "section"]]
# insert features in dataframe
for i in flist:
    featuresdf.loc[:, i] = Aheader["section"].apply(
        lambda st: i in " ".join([stemmer.stem(i) for i in st.split(" ")])
    )  # st.lower()
# insert expected results in dataframe
for i in conca.index.values:
    featuresdf.insert(
        0, i, Aheader["section_type"].apply(lambda st: st.lower() == i.lower())
    )
# reset index of dataframe
featuresdf.reset_index(drop=True)

##generate features/expected-classification-result dataframe
featdf = Pheaders[["section_type", "section"]]
# insert features in dataframe
for i in flist:
    featdf.loc[:, i] = Pheaders["section"].apply(
        lambda st: i in " ".join([stemmer.stem(i) for i in st.split(" ")])
    )  # st.lower()
# insert expected results in dataframe
for i in concap.index.values:
    featdf.insert(
        0, i, Pheaders["section_type"].apply(lambda st: st.lower() == i.lower())
    )
# reset index of dataframe
featdf.reset_index(drop=True)

###prediction

# splitting data set to training_set and test_set
training_set2 = featdf.iloc[0 : int(len(featdf) * 0.66), :]
test_set2 = featdf.iloc[int(len(featdf) * 0.66) :, :]  # len(featdf)

# initialise the result dataframe
class_result2 = test_set2[["section_type", "section"]].reset_index(drop=True)
class_result2.loc[:, "estimated_type"] = "None"

# make estimations
for i in range(0, len(concap)):
    X22 = featdf.iloc[:, len(concap) + 2 : len(training_set2.columns)].values
    Y22 = featdf.iloc[:, i].values

    # feature scaling
    sc_X = StandardScaler()
    X22 = sc_X.fit_transform(X22)
    # predict the test set result
    Y_pred22 = classifier21.predict(X22)

    # store perdiction for class i in the result dataframe
    Y_true2 = class_result2.index[Y_pred22 == True]
    for tr_ind in Y_true2:
        if class_result2.loc[tr_ind, "estimated_type"] == "None":
            class_result2.loc[tr_ind, "estimated_type"] = test_set2.columns[i]


# test the first classifier from new document

# select a html document;output:fname

from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw()
fname = askopenfilename(
    title="Ouvrir votre document",
    filetypes=[("html files", ".html"), ("all files", ".*")],
)
##
parser_test = doc_prep(fname)
parser_test = feature_application(parser_test)
doc_prop = parser_test.iloc[:, 3 : len(parser_test.columns)].values
Y = parser_test.iloc[:, 0].values
headers_predict = classifier1.predict(doc_prop)

# confusion Matrix
cm = confusion_matrix(Y, headers_predict)
print("confusion_matrix:\n", cm)
# accuracy
print("accuracy = ", accuracy_score(Y, headers_predict))
# recall
print("recall = ", recall_score(Y, headers_predict))
# precision
print("precicion = ", precision_score(Y, headers_predict))
