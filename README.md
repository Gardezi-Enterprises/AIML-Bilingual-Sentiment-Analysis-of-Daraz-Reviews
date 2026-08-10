# Bilingual Sentiment Analysis — Code-Mixed Urdu/English Reviews

Sentiment classification for code-mixed Urdu-English product reviews, where English-only sentiment models fail.

Reviews on South Asian e-commerce platforms are routinely written in Roman Urdu, English, or both within a single sentence. Off-the-shelf sentiment tools score these as noise. This model reads them.

## Approach

| Stage | Implementation |
|---|---|
| Data | Two annotated review sets merged — labelled bilingual reviews and code-mixed Urdu-English reviews |
| Class balancing | Undersampling of the majority class to prevent a degenerate classifier |
| Preprocessing | Lowercasing, tokenisation, and stopword removal against **both** an English list and a custom 500-word Urdu list (`Urdustopwords.txt`) |
| Features | TF-IDF vectorisation |
| Model | Support Vector Classifier, tuned on the balanced set |
| Serving | Vectoriser and model serialised with pickle, loaded by a Streamlit app for real-time scoring |

The bilingual stopword handling is the part that matters — removing English stopwords alone leaves Urdu function words dominating the TF-IDF space and washing out sentiment signal.

## Stack

Python · scikit-learn · pandas · Streamlit · Matplotlib

## Running It

```bash
pip install -r requirements.txt
streamlit run main.py
```

Paste a review in any mix of Urdu and English; the app returns a sentiment classification.

## Repository Contents

| File | Purpose |
|---|---|
| `main.py` | Streamlit scoring interface |
| `Jupyter file.ipynb` | Data merging, preprocessing, training and evaluation |
| `svcModel(ok).pickle` | Trained SVC classifier |
| `vectorizer.pickle` | Fitted TF-IDF vectoriser |
| `Urdustopwords.txt` | Custom Urdu stopword list |
| `*.csv` | Annotated review datasets |

## Extending It

The serialised model and vectoriser are the whole inference path — wrapping them in a FastAPI endpoint is a few lines, and the same preprocessing applies to any Roman-Urdu text domain, not just product reviews.
