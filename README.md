# 📊 Bilingual Sentiment Analysis of Daraz Reviews

## 📝 Project Overview
This project analyzes **bilingual product reviews (Urdu + English)** from **Daraz.pk**, classifying them into **positive**, **negative**, or **neutral** sentiments using supervised machine learning techniques. It incorporates traditional text preprocessing, TF-IDF vectorization, and a **Flask-based web interface** for real-time sentiment prediction.

---

## 🧰 Technologies and Tools Used

| Technology         | Purpose                                      |
|--------------------|----------------------------------------------|
| Python             | Core programming language                    |
| Pandas             | Data loading and manipulation                |
| scikit-learn       | ML algorithms, TF-IDF vectorization          |
| Matplotlib         | Data visualization                           |
| Flask              | Web application framework                    |
| Pickle             | Model serialization                          |
| Jupyter Notebook   | Exploratory analysis & model development     |
| HTML/CSS (via Flask)| Frontend of web UI                          |
| Urdu Stopwords List| Urdu text preprocessing                      |

---

## 📦 Dataset

Two annotated datasets are used:

- `Daraz FYP Dataset.csv` – Labeled bilingual reviews.
- `daraz-code-mixed-product-reviews.csv` – Urdu-English code-mixed reviews.

### 🔧 Preprocessing Steps
- Urdu & English stopword removal (custom list included)
- Dataset balancing via **undersampling**
- TF-IDF vectorization for feature extraction

---

## 📁 Project Structure

Bilingual Daraz Review Classifier/
├── App/
│ ├── main.py # Flask backend
│ ├── svcModel(ok).pickle # Trained SVC model
│ ├── vectorizer.pickle # Trained TF-IDF vectorizer
│ └── how_to_run_web_app.txt # Instructions to run the app
├── Daraz FYP Dataset.csv
├── daraz-code-mixed-product-reviews.csv
├── Urdustopwords.txt # Custom Urdu stopword list
├── readme.txt # Basic explanation
├── Jupyter file.ipynb # Analysis and model training
├── Final Report.pdf # Academic project report


---

## 🧠 Model Development

All training and analysis steps are documented in `Jupyter file.ipynb`:

1. Load and merge both datasets  
2. Label balancing using undersampling  
3. Text preprocessing  
    - Lowercasing  
    - Stopword removal (English & Urdu)  
    - Tokenization  
4. Feature extraction using **TF-IDF**  
5. Training a **Support Vector Classifier (SVC)**  
6. Saving model and vectorizer with **Pickle**  

---

## 🌐 Web Application

An interactive Flask web app allows users to input a product review and receive a **sentiment prediction**.


### How to Use
Clone or download this repository

Ensure Python 3 and required libraries are installed:

scikit-learn

pandas

flask

Run Jupyter file.ipynb to explore and (re)train the model

Launch the web app using the main.py script in the App/ directory


### ▶️ How to Run

  ```bash
  cd App
  python main.py

  Then open your browser at http://127.0.0.1:5000/
