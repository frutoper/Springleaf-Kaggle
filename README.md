# Springleaf-Kaggle

Kaggle is a website that hosts online data science competitions.  The goal of this competition is to identify consumers who are most interested in a direct mail offer from Springleaf. Participants mine anonymized customer information.  For a full description click [here](https://www.kaggle.com/c/springleaf-marketing-response).

I finished 445 out of 2225 participants.  This was good for a "Top 25%" honor from Kaggle.  In this competition I experimented with a few different machine learning algorithms.  I found the gradient boosting to be the best.  

I used the XGBoost R library and spent some time finding the best hyperparameters. The methodology I used was Random Search.  I was convinced to use Random Search over grid search by [this blog post on medium](https://medium.com/rants-on-machine-learning/smarter-parameter-sweeps-or-why-grid-search-is-plain-stupid-c17d97a0e881).

I separated the Kaggle training set into a smaller training set and a validation set.  This ensured my results wouldn't overfit to the public leaderboard.  In the end my final submission position was one place better than my public leadboard position.
