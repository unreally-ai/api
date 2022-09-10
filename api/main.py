# -------- API ----------
from fastapi import FastAPI

# -------- News API ----------
from rapidapi_key import x_rapidapi_key

# ------- Machine Learning ---------
import pandas as pd

import torch
import torch.nn as nn

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity

from nltk.stem import WordNetLemmatizer

# ---------------- LETS GET TO THE CODE ---------------------

# Create News API Client
URL = "https://rapidapi.p.rapidapi.com/api/search/NewsSearchAPI"
HEADERS = {
    "x-rapidapi-host": "contextualwebsearch-websearch-v1.p.rapidapi.com",
    "x-rapidapi-key": x_rapidapi_key
}

# ----------------------------- PARAMETERS -----------------------------
CUSTOM_SW = ["semst","u"] # TODO: add to actual stopwords
VOCAB_PATH = "/Users/vince/unreally/helpers-main/vocab_script/vocab_headlines.csv"
BODY_PATH = "/Users/vince/unreally/helpers-main/vocab_script/vocab_bodies.csv"
VOCAB = "/Users/vince/unreally/helpers-main/vocab_script/kowalsky_vocab.csv"
USE_LEMMATIZER = True

# TODO: select headlines & body out of dataset 
vocab_df = pd.read_csv(VOCAB, header=None)

# ----------------------------- BOW VECTORIZER PIPELINE -----------------------------
# takes [string], returns lowercased & lemmatized [string]
def lem_str(in_string):
    out_str = [""]
    lemmatizer = WordNetLemmatizer()
    for word in in_string.split():
        out_str[0] += (lemmatizer.lemmatize(word.lower()) + " ")
    
    return out_str

# takes vocab & returns bow_vectorizer
def load_vectorizer(path):
    # define stopwords
    sw = text.ENGLISH_STOP_WORDS.union(["book"])
    # read vocabulary
    vocab_df = pd.read_csv(path, header=None)
    vocab_df = vocab_df.drop([0])

    bow_vectorizer = CountVectorizer(
        stop_words=sw,
        max_features=5000,
        vocabulary=vocab_df[1]
    )


    return bow_vectorizer

# takes string & path to vocab and yeets it through the BoW pipeline
def create_bow(in_string, path):
    bow_vectorizer = load_vectorizer(path)
    if USE_LEMMATIZER == True:
        bow = bow_vectorizer.fit_transform(lem_str(in_string), y=None)
    else:
        bow = bow_vectorizer.fit_transform([in_string], y=None)

    return bow


# ----------------------------- TF-IDF PIPELINE -----------------------------

def create_tf(bow_vec):
    tfreq_vec = TfidfTransformer(use_idf=False).fit(bow_vec)
    tfreq = tfreq_vec.transform(bow_vec)

    return tfreq

# TODO figure out

def create_tfidf(bow):
    tfreq_vec = TfidfTransformer(use_idf=True)
    tfreq = tfreq_vec.fit_transform(bow)

    return tfreq

# -------------- GENEREATE 10001 VECTOR --------------
def yeet2vec(head, body):

    # get our sub-vectores
    claim_tf = create_tf(create_bow(head, VOCAB))
    body_tf = create_tf(create_bow(body, VOCAB))
    
    claim_tfidf = create_tfidf(create_bow(head, VOCAB))
    body_tfidf = create_tfidf(create_bow(body, VOCAB))

    print("  - created sub-vectors ✅")

    # tasty cosine similarity
    c_sim = cosine_similarity(claim_tfidf ,body_tfidf)

    # do the yeeting
    # HERE IS SOME ERRROR WITH CONCAT
    claim_df = pd.DataFrame(claim_tf.toarray()) 
    body_df = pd.DataFrame(body_tf.toarray())
    c_sim_df = pd.DataFrame(c_sim)
    
    tenk = pd.concat([claim_df,c_sim_df,body_df],axis=1)
    tenk = tenk.to_numpy()
    tenk = torch.from_numpy(tenk)
    terror = [tenk]

    print("  - created 10k vector ✅")
    return tenk

# load the model
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# set dimensions
in_dim = 10001
hidden_dim = 100
out_dim = 4

class NN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(NN, self).__init__()
        # define layers
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_dim, out_dim)
    
    # applies layers with sample x
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NN(in_dim, hidden_dim, out_dim)
model.load_state_dict(torch.load('api/kowalsky_72_balanced.pth', map_location=torch.device('cpu')))
model.eval()

def predict(tenk_vec):
    with torch.no_grad():
        pred = model(tenk_vec.float())
        pred_out, pred_idx = torch.max(pred, 1)


    classes = ['agree', 'disagree', 'discuss', 'unrelated']
    print(pred)
    stance = classes[((pred_idx.data).numpy()[0])]
    return dict(zip(classes, pred.tolist()[0])), stance

app = FastAPI()

@app.get("/")
async def route():
    return {"message":"Willkommen zu der Unreally API"}

@app.get("/predict")
async def use_model(query: str):
    # get bodies matching test_string
    test_string = query
    # get bodies matching test_string
    page_number = 1
    page_size = 5
    auto_correct = True
    safe_search = True
    with_thumbnails = False
    from_published_date = ""
    to_published_date = ""

    querystring = {"q": test_string,
                "pageNumber": page_number,
                "pageSize": page_size,
                "autoCorrect": auto_correct,
                "safeSearch": safe_search,
                "withThumbnails": with_thumbnails,
                "fromPublishedDate": from_published_date,
                "toPublishedDate": to_published_date}

    response = requests.get(URL, headers=HEADERS, params=querystring).json()
    test_body = "" # initiate test_body (the body we need to test the stance against the test_string)
    sources = []
    print(len(response['value']))
    for article in response['value']:
        body = article['body']
        sources.append(article['url'])
        test_body += body
        vector = yeet2vec(test_string, test_body)
        prediction = predict(vector)
    if len(test_body) == 0:
        return "No articles found :/"
    # yeets strings through pipeline, outputs finished 10k vector
    print(len(test_body))
    print(test_body)
    vector = yeet2vec(test_string, test_body)
    values, prediction = predict(vector)
    # send answer tweet
    return {'prediction': prediction,'values': values, 'sources': sources}