# Sentiment Analysis
**Georgia Institute of Technology CS4641 Team 7**

Adil Farooq  
Aditya Singh  
Jiaying Zhu  
Ngoc Khai Huyen Nguyen  
Yongyu Qiang  

## Introduction/Background
Sentiment analysis is a NLP field that identifies, extracts, and categorizes subjective information from text. Among others, naive Bayes, K-nearest neighbors, decision trees, support vector machines, and neural networks are popular ML algorithms in literature used for textual analysis [1]. Specifically, Maks and Vossen showed that lexicon models that identify parts of speech and their relationships are effective for sentiment analysis [2].

## Dataset
We’ll categorize news as positive, negative, or neutral using <a href="https://huggingface.co/datasets/fancyzhx/ag_news">AG's news topic classification dataset </a>. It contains 120k news headlines from Hugging Face tagged with topics like World, Sports, Business, or Science/Tech from ~2000 sources.


## Problem Definition
With exponential growth in textual data generated daily, there is a pressing need for sentiment analysis to efficiently process this vast amount of information. Automated sentiment analysis enables agencies to identify emerging trends and gain insights into how information is perceived by the public. The main challenge lies in developing accurate and robust models that can effectively understand and interpret nuanced opinions expressed in text data.

## Methods
### Data Preprocessing
For preprocessing, we’ll transform the dataset into Bag-of-Words representations using CountVectorizer from Sklearn. Sentences will be simplified into individual word tokens and converted to lowercase to help with analysis. Using NLTK, stop words [3] like common articles, prepositions, etc. that do not contribute to meaning will be removed. Verb conjugations or suffixes will be removed with the PorterStemmer algorithm. This reduces redundant complexity, allowing variations like "run", “Run”, “run the”, and "running" to be treated similarly. Additionally, we’ll use principal component analysis to decrease the number of unique words and reduce dimensionality.

The lexicon model approach in [2] is a viable alternative for preprocessing. As Uysal and Gunal [3] discovered, testing various combinations of the above options could also significantly impact text classification algorithm performance.

### Machine Learning Algorithms
For supervised machine learning algorithms, we plan to use naive Bayes, a relatively fast yet effective classifier [4] as a baseline for comparison. We’ll also use Support Vector Machines, as they remain effective even with high-dimensional data [5], which will help with the Bag-of-Words representation. Additionally, Convolutional Neural Networks can be used as they’re flexible and may capture more complex relationships that we might not have previously considered. For unsupervised learning, k-means can be used as a trial run to see if naively clustering based on distances will reveal any meaningful information regarding various sentiments.

## Potential Results and Discussion
In sentiment analysis/classification, we typically use traditional binary classification metrics such as F1-score, precision, recall, and accuracy [6], especially in multiclass scenarios where achieving a high macro-F1 score and accuracy is crucial [7].

Precision, recall, and F1 scores can provide per-class insights, while macro averaging prioritizes balanced consideration across classes [6]. 

Our goal is to achieve high accuracy and macro-F1 scores between 0.93 and 0.97 [7,8]. We’ll also use confusion matrices to visually display results and make our discussion of the results more intuitive and investigate whether the AUC metric provides any useful insights. [6] These metrics will guide adjustments to our methods and models, ensuring robust performance evaluation.




## Gantt Chart
[Team 7-Gantt Chart](https://gtvault-my.sharepoint.com/:x:/g/personal/jzhu491_gatech_edu/ETq22_plETFBv1rCJ58wni0B9rK3IZXenrD21P_5AH10gw?e=c3O0JP)

## Contributions
| Name                          | Proposal Contributions |
| -----------                   | ----------- |
| Adil Farooq                   | Potential Results & Discussion, GitHub Page       |
| Aditya Singh                  | Potential Dataset, Video Creation & Recording        |
| (Jenny) Jiaying Zhu           | Problem Definition, Methods, Potential Dataset        |
| (Khai) Ngoc Khai Huyen Nguyen | Introduction & Background, Problem Definition, GitHub Page|
| (Frank) Yongyu Qiang          | Introduction & Background, Methods, Video Creation & Recording        |


## References
[1] Michał Mirończuk, Highlights, “A recent overview of the state-of-the-art elements of text classification,” Expert Systems with Applications, https://www.sciencedirect.com/science/article/pii/S095741741830215X (accessed Jun. 14, 2024).

[2] Maks et al., “A lexicon model for deep sentiment analysis and opinion mining applications,” Decision Support Systems, https://www.sciencedirect.com/science/article/pii/S0167923612001364 (accessed Jun. 14, 2024).

[3] Alper Kursat Uysal et al., “The impact of preprocessing on text classification,” Information Processing & Management, https://www.sciencedirect.com/science/article/pii/S0306457313000964 (accessed Jun. 14, 2024).

[4] “1.9. naive Bayes,” scikit, https://scikit-learn.org/stable/modules/naive_bayes.html (accessed Jun. 14, 2024).

[5] “1.4. Support Vector Machines,” scikit, https://scikit-learn.org/stable/modules/svm.html (accessed Jun. 14, 2024).

[6]“3.3. Metrics and scoring: quantifying the quality of predictions — scikit-learn 0.22.1 documentation,” scikit-learn.org. https://scikit-learn.org/stable/modules/model_evaluation.html

[7]“Which metrics are used to evaluate a multiclass classification model’s performance?,” www.pi.exchange. https://www.pi.exchange/knowledgehub/metrics-to-consider-when-evaluating-a-multiclass-classification-models-performance (accessed Jun. 15, 2024).

[8]“Accuracy, precision, and recall in multi-class classification,” www.evidentlyai.com. https://www.evidentlyai.com/classification-metrics/multi-class-metrics



