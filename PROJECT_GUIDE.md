# Project Guide — Ad Click Prediction (A to Z, for beginners)

> This file is a personal/internal explainer — not the public README. It walks through everything about this project so anyone new can understand it fully.

---

## 1. What is this project? (In plain English)

Imagine an ad company shows banner ads to 1000 people. Some people click, some don't. This project takes data about those 1000 people — their age, income, how much time they spend online, etc. — and builds a model that **predicts, for a brand-new person, whether they will click the ad or not (Yes/No)**.

That's it. No deep learning, no neural networks, no fancy buzzwords — just a clean, classic machine learning pipeline: **load data → explore it → clean it → train a simple model → check how good it is**.

---

## 2. Why this project?

- **Real-world relevance:** Ad click prediction is literally what powers digital advertising (Google Ads, Facebook Ads, etc. all do a bigger version of this).
- **Learning goal:** It's a great starter project to practice the *full* ML workflow — EDA, preprocessing, model training, evaluation — without getting lost in complexity.
- **Business angle:** Beyond just predicting clicks, the project also looks at *which* features (age? income? time on site?) actually drive clicks — useful for marketing decisions.

## 3. Is this a simple project? Yes — and that's the point

| It IS | It is NOT |
|---|---|
| A single CSV file with 1000 rows | Big Data / streaming pipeline |
| One algorithm: Logistic Regression | Deep Learning / Neural Networks |
| Runs on a laptop in seconds | Needs a GPU or cloud cluster |
| A Jupyter Notebook you can read top-to-bottom | A production deployed API/service |

If you're new to ML, this is intentionally a **"Hello World" of classification** — small enough to understand every line, but realistic enough to look good on a resume.

---

## 4. Tech Stack — what's used and why

| Tool | Role in this project | Why this and not something else |
|---|---|---|
| **Python** | Programming language | Standard for data science |
| **Pandas** | Load & manipulate the CSV data | Easiest way to work with tabular data |
| **NumPy** | Numerical operations | Pandas/Scikit-learn are built on it |
| **Matplotlib / Seaborn** | Charts (histograms, heatmap, pairplot) | Best combo for quick statistical plots |
| **Scikit-learn** | `train_test_split`, `StandardScaler`, `LogisticRegression`, metrics | The go-to library for classic ML — simple API, no overkill |
| **Jupyter Notebook** | Where all the code lives | Lets you run code + see plots + write notes step by step, great for learning |

No web framework, no database, no deployment tooling is used — this is a **modeling/analysis project**, not a deployed app (see [Future Scope](#11-future-scope--what-could-be-added-next) for that).

---

## 5. The Dataset

File: `data/advertising.csv` — 1000 rows, 10 columns.

| Column | Meaning |
|---|---|
| `Daily Time Spent on Site` | Minutes the user spends on the site per visit |
| `Age` | User's age |
| `Area Income` | Average income of the user's area |
| `Daily Internet Usage` | Minutes the user spends on the internet daily |
| `Ad Topic Line` | Headline text of the ad shown (text, not used numerically here) |
| `City` | User's city |
| `Male` | 1 = Male, 0 = Female |
| `Country` | User's country |
| `Timestamp` | When the ad was shown |
| `Clicked on Ad` | **Target column** — 1 = clicked, 0 = did not click |

Only the 5 numeric/behavioral columns (`Daily Time Spent on Site`, `Age`, `Area Income`, `Daily Internet Usage`, `Male`) are used to predict `Clicked on Ad`. Text columns (`Ad Topic Line`, `City`, `Country`, `Timestamp`) are not fed into the model in this version.

---

## 6. Project Structure

```text
ad-click-prediction/
│
├── data/
│   └── advertising.csv          # The raw dataset (1000 rows)
│
├── notebook/
│   └── ad_click_model.ipynb     # ALL the code: EDA → preprocessing → model → evaluation
│
├── images/                       # Charts saved by the notebook (used in README.md)
│   ├── heatmap.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── pairplot.png
│
├── requirements.txt              # Exact Python packages needed to run this project
├── README.md                     # Public-facing project summary
└── PROJECT_GUIDE.md              # This file — full internal explainer
```

There is intentionally **no `src/` folder, no separate script files, no config files** — everything lives in one notebook because the project is small enough that splitting it into modules would add complexity without adding value.

---

## 7. How it works — step by step

This mirrors exactly what `notebook/ad_click_model.ipynb` does, in order:

1. **Load the data** — read `advertising.csv` into a Pandas DataFrame.
2. **Explore it (EDA)** — look at `.info()`, `.describe()`, histograms of Age/Time Spent/Income, and a countplot of how many people clicked vs. didn't.
3. **Look for relationships** — jointplots (Age vs. Income, Age vs. Time Spent, etc.) and a **pairplot** colored by whether the user clicked, to visually spot patterns.
4. **Correlation heatmap** — a numeric summary of how strongly each feature relates to every other feature (and to the target).
5. **Prepare the data for modeling**
   - Pick the 5 useful numeric features (`X`) and the target column (`y`).
   - **Split** into 70% training data / 30% test data — the model never sees the test data while learning, so testing on it tells us how it'll perform on brand-new users.
   - **Scale** the features with `StandardScaler` (makes every feature have similar range/units — logistic regression converges better and treats features fairly). Important: the scaler is *fit* only on training data and just *applied* to test data — this avoids "cheating" by leaking test-set information into training.
6. **Train the model** — a `LogisticRegression` classifier. Despite the name, it's used here for **classification** (click / no-click), not predicting a continuous number. It outputs a probability between 0 and 1, which is turned into a Yes/No using a 0.5 threshold.
7. **Evaluate the model** on the untouched test data:
   - **Confusion Matrix** — how many predictions were right/wrong, broken down by class.
   - **Classification Report** — precision, recall, F1-score.
   - **ROC Curve & AUC** — how well the model separates clickers from non-clickers across all possible thresholds (not just 0.5).
   - **Accuracy** — the overall percentage of correct predictions.

---

## 8. Results — and what they actually mean

| Metric | Score | In plain English |
|---|---|---|
| **Accuracy** | **97%** | Out of 100 test users, the model correctly predicted click/no-click for ~97 of them |
| **AUC (ROC)** | **~0.986** | Near 1.0 means the model is excellent at ranking "likely to click" users above "unlikely to click" ones — very close to a perfect separator |

Why so high? This dataset has very clear behavioral patterns (e.g., people who spend *less* time on the internet tend to click ads *more* — likely because they're less "internet-savvy" and less ad-blind). That clean separation is visible directly in the pairplot (`images/pairplot.png`).

**Honest caveat:** this is a single 70/30 split, not cross-validated, and the dataset is small/clean (likely semi-synthetic, from a well-known Kaggle "Advertising" dataset). Real-world ad data is messier — treat 97% as "this dataset is easy," not "logistic regression is magic."

---

## 9. How to run it yourself

```bash
git clone <this-repo-url>
cd ad-click-prediction
pip install -r requirements.txt
jupyter notebook notebook/ad_click_model.ipynb
```

Then just run all cells top to bottom (`Run All` in Jupyter). The dataset path is already relative (`../data/advertising.csv`), so it works out of the box — no Google Drive, no path editing.

---

## 10. Limitations (being honest)

- Only tested with **one train/test split** — no cross-validation, so the 97% number could vary slightly with different splits.
- Only **one algorithm** tried (Logistic Regression) — no comparison against Random Forest, XGBoost, etc.
- **Text/categorical columns ignored** (`City`, `Country`, `Ad Topic Line`) — they might hold useful signal that's currently thrown away.
- The trained model is **not saved** (no `.pkl` file) and there's **no script or API** to use it on new data outside the notebook — it's an analysis notebook, not a deployed product.
- No handling for class imbalance, outliers, or missing values beyond a basic check (this dataset happens to have none).

## 11. Future Scope — what could be added next

- [ ] **Cross-validation** (e.g., 5-fold) for a more reliable accuracy estimate.
- [ ] **Try other models** — Random Forest, XGBoost, SVM — and compare.
- [ ] **Use categorical features** — encode `Country`/`City`, or extract hour-of-day from `Timestamp`.
- [ ] **Hyperparameter tuning** with `GridSearchCV`.
- [ ] **Explainability** — SHAP or feature-importance plots to clearly rank "what drives clicks."
- [ ] **Save the trained model** (`joblib`/`pickle`) so it can be reused without retraining.
- [ ] **Wrap it as an API** (FastAPI/Flask) so a real app could send user data and get a click-probability back.
- [ ] **Simple web demo** (Streamlit) — type in Age/Income/etc. and see the predicted click probability live.

---

## 12. Author

Tanu Sharma
