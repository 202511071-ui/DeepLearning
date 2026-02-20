Lab01 : Imitating Fakespot
202511071
Trush Chauhan
Lab02 : This project applies pretrained GloVe (100‑dimensional) word embeddings to two predictive tasks on movie metadata:

Regression – predicting the voting_average (rating) from a single text column.

Multi‑label classification – predicting movie genres from a single text column.

Key Steps
Data Preparation: Cleaned and split the dataset (70/15/15) using only overview, tagline, keywords, genres, and vote_average.

Embedding Generation: Used TF‑IDF weighted averaging of GloVe vectors to produce 100‑dimensional document embeddings for each text column. Coverage on the training set was ~90%.

Regression (Rating Prediction): Trained a neural network on overview embeddings. The model achieved MSE = 2.2983 and RMSE = 1.5160, underperforming the baseline (global mean) which gave MSE = 1.2825.

Multi‑label Classification (Genre Prediction): Used a OneVsRest classifier with logistic regression on the embeddings. The model using overview achieved Micro‑F1 = 0.4773, Macro‑F1 = 0.2999, and Hamming Loss = 0.1018, significantly outperforming the tagline‑based model (Micro‑F1 = 0.3011). This confirms that longer, richer text (overview) contains stronger genre signals.

Text Analysis:

Word frequency per genre (top 10 most common words, and bottom words with frequency ≥ 3).

Genre‑indicative words extracted from logistic regression coefficients, showing which words are most positively associated with each genre.

Conclusion
GloVe embeddings, combined with TF‑IDF weighting, are effective for genre classification but less so for rating prediction, which likely requires additional features beyond textual content.

The project demonstrates the complete pipeline from data preprocessing to model training and interpretability analysis using pretrained embeddings.

