# -------- Twitter API ----------
import tweepy
import keys

# -------- News API ----------
import requests

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
    "x-rapidapi-key": keys.rapidapi_key
}

# Create Twitter API Client
client = tweepy.Client(
    consumer_key=keys.API_KEY,
    consumer_secret=keys.API_KEY_SECRET,
    access_token=keys.ACCESS_TOKEN,
    access_token_secret=keys.ACCESS_TOKEN_SECRET)

# ----------------------------- PARAMETERS -----------------------------
CUSTOM_SW = ["semst","u"] # TODO: add to actual stopwords
VOCAB = "kowalsky_vocab.csv"
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
model.load_state_dict(torch.load('kowalsky_72_balanced.pth', map_location=torch.device('cpu')))
model.eval()

def predict(tenk_vec):
    with torch.no_grad():
        pred = model(tenk_vec.float())
        pred_out, pred_idx = torch.max(pred, 1)


    classes = ['agree', 'disagree', 'discuss', 'unrelated']
    print(pred)
    stance = classes[((pred_idx.data).numpy()[0])]
    return dict(zip(classes, pred.tolist()[0])), stance

def use_model(query: str):
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
    is_safe = response['value'][0]['isSafe']
    # send answer tweet
    return values, prediction, sources, is_safe

import time

while True:
    last_time = open("time.txt", "r")
    start_from = last_time.read()
    # get @calctruth mentions
    tweets = client.get_users_mentions(user_auth=True,id=keys.USER_ID,expansions=["referenced_tweets.id"], since_id=start_from)
    last_time.close()

    try:
        for tweet in tweets.data:
            # Handle mention and get the tweet we need to evaluate
            respond_to = tweet.id # the tweet which mentioned @calctruth, and the tweet we need to respond to
            reference = tweet.referenced_tweets[0]['id'] # the id of the tweet @calctruth was mentioned under
            reference_tweet = client.get_tweet(user_auth=True,id=reference) # the tweet @calctruth was mentioned under
            test_string = reference_tweet[0]['text'] # the tweet we will evaluate
            print(test_string)
            values, prediction, sources, is_safe= use_model(test_string)
            output = f"Predicted as: {prediction}, with an accuracy of 72.5%."
            # send answer tweet
            if is_safe:
                response = client.create_tweet(user_auth=True,text=output+" One of the sources: "+sources[0], in_reply_to_tweet_id=respond_to)
                print("tweet was successful")
            else:
                response = client.create_tweet(user_auth=True,text=output, in_reply_to_tweet_id=respond_to)
            '''
            except:
                print("No article found")
                response = client.create_tweet(user_auth=True,text="Sorry, I couldn't find any articles concerning this topic 😔", in_reply_to_tweet_id=respond_to)
                print(response)
            '''
            last_time = open("time.txt", "w")
            last_time.write(f"{respond_to}")
            last_time.close()
            print(respond_to)
            print(response)
    except:
        print(f"You have not been mentioned since: id={start_from}")
    time.sleep(10)