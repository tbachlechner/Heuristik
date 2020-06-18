<p align="left">
<img src=./data/heuristik_logo.jpeg width="350"/>
</p>

## Beating trends. News indicative of imminent stock price volatility.

Heuristik is a deep learning pipeline that predicts business news items that are indicative of large future stock price changes. The model is deployed at [heuristik.me](http://www.heuristik.me).

Heuristik periodically parses thousands of business news items and infers a score that reflects how likely the stock price of the associated company will fluctuate by >5% over the following three days.

### Pipeline and model
The complete pipeline is shown in the schematic.
<p align="center">
<img src=./data/schematic.jpg width="850"/>
</p>

Heuristik collects historical and current news and stock prices from the [StockNews](https://stocknewsapi.com) and [AlphaVantage](https://www.alphavantage.co) APIs, respectively.

The historical text and time-series data is stored, cleaned and labeled according to whether the stock price changes significantly using the [triple-barrier method](https://mlfinlab.readthedocs.io/en/latest/implementations/tb_meta_labeling.html). To clean the news data it is important to remove the unique company name, which improves inference precision (e.g. replace "Tesla", "Apple", ... with the word "company"). The resulting dataset is imbalanced. About 10% of the news reports are followed by a significant change in stock price (i.e. >5% fluctuation over 3 days). 

The model consists of a pre-trained [Huggingface BERT](https://github.com/huggingface/transformers) transformer model, supplemented by the [ReZero architecture](https://arxiv.org/pdf/2003.04887.pdf). Since the dataset is imbalanced, the [FocalLoss](https://arxiv.org/pdf/1708.02002.pdf) function is used, which emphasizes the wrong predictions of the model and significantly improves performance on the validation and test-sets. The model is trained using the AdamW optimizer.

The trained model is used to infer the relevancy score on current news items. The package Streamlit is used to deploy the results, and to interface new inquiries with the inference and data-collection threads.

### Evaluation

The model selected out of the ~2500 news items in the test set 50 strong predictions. 76% those 50 news reports were indeed followed by a significant change in the price of the associated stock. To compare the model, a plain Huggingface Transformer model and a 10-layer fully connected network were trained on the same data, achieving about 70% and 10% precision on the test-set respectively. The ReZero-Transformer significantly outperforms the plain Transformer, and (as expected) a fully connected network performs no better than a random guess on this task. Each of the models were trained for 10 epochs, with a batch size of 32 news items, and each news item was trimmed to a maximum of 100 words. The figure shows the precision on the validation set for each of the three architectures.

<p align="center">
<img src=./data/learning_curve.jpg width="550"/>
</p>

### Issues/Notes

- Over-fitting is one of the main issues. The dataset used to train the model contains about 5M words, and the model achieves a >99% accuracy on the training set after about 10 epochs, inhibiting any further increase in generalization performance. While hyperparameter-tuning might somewhat ammeliorate this issue, a larger dataset is likely most effective to address this issue.

- News reports containing "earnings calls" are predicted as highly significant.

- Recall, F1 score and accuracy each are not reliable metrics to evaluate the performance of the model. Recall and F1 are unreliable because they both assume that the label actually is predictable by the data. This is not the case for the problem at hand: many large changes in the stock price are either not reported in news items, or are entirely independent of the company and rather driven by overall market sentiment. Accuracy is a poor measure because the dataset is highly imbalanced. For these reasons, precision is chosen as the main performance indicator.
