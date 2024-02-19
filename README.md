# Stock Price Prediction from Central Bank Speeches

## Overview
This project focuses on predicting stock price movements using speeches from central banks, namely the Federal Reserve (FED) and the European Central Bank (ECB). The objective is to assist financial institutions, such as Natixis Corporate & Investment Banking, in making informed decisions regarding stock market investments.

## Objectives
The primary goals of this project are as follows:
- Predicting the movement of financial market indices, specifically VIX and EURUSD 1M.
- Leveraging Natural Language Processing (NLP) techniques to analyze central bank speeches.
- Providing insights to aid Natixis in their decision-making process regarding stock market actions.

## Data
The dataset comprises JSON files containing historical stock prices and corresponding speeches from central banks over the past decade. These data are provided by Natixis to facilitate analysis and decision-making in the area of stock market investments.

## Methodology
1. **Data Preprocessing:** Convert speeches into a suitable format for NLP analysis. Remove non-English speeches and perform text cleaning.
2. **Feature Engineering:** Extract features from speeches using TF-IDF (Term Frequency-Inverse Document Frequency) weighting.
3. **Modeling:** Apply Machine/Deep Learning algorithms such as Random Forest for classification and Stacked LSTM for regression.
4. **Evaluation:** Assess model performance using RMSE for regression and accuracy for classification.

## Conclusion
This project serves as a demonstration of the potential utility of NLP techniques in analyzing central bank speeches for predicting stock price movements. By providing actionable insights, it aims to support financial institutions like Natixis in making informed decisions regarding their investments in the stock market.
