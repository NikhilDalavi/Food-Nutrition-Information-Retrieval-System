from flask import Flask, render_template, request, jsonify
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

app = Flask(__name__)

df = pd.read_csv("food_nutrition.csv")
df = df.fillna("")

name_col = "Dish Name"
cal_col = "Calories (kcal)"
protein_col = "Protein (g)"


stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)  # Tokenization
    
    tokens = [
        stemmer.stem(word) for word in tokens
        if word not in stop_words and word not in string.punctuation
    ]
    
    return " ".join(tokens)

df["processed_name"] = df[name_col].astype(str).apply(preprocess)

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(df["processed_name"])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    q = request.form["query"]
    
    q_processed = preprocess(q)
    
    sims = cosine_similarity(vectorizer.transform([q_processed]), tfidf).flatten()
    idxs = sims.argsort()[::-1][:15]

    out = []
    for i in idxs:
        if sims[i] > 0:
            item = df.iloc[i].to_dict()
            item["id"] = int(i)
            item["score"] = round(float(sims[i]), 2)
            out.append(item)

    return jsonify(out)

@app.route("/details/<int:item_id>")
def details(item_id):
    item = df.iloc[item_id].to_dict()
    return jsonify(item)

@app.route("/weightloss")
def weightloss():
    d = df.sort_values(by=[cal_col, protein_col], ascending=[True, False]).head(15)
    return jsonify(d.to_dict(orient="records"))

@app.route("/weightgain")
def weightgain():
    d = df.sort_values(by=[cal_col, protein_col], ascending=[False, False]).head(15)
    return jsonify(d.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True)