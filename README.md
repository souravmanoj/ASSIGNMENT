Report on Trained ALBERT Language Model
By: Sourav Manoj
22033008
1 Introduction
1.1 A Brief Overview of the ALBERT Model
Language models (LMs) play the key role in generating
messages in many currently most popular NLP systems.
Their purpose is to comprehend, generate, and manipulate
human language. The more recent development of
transformer-based models such as BERT and ALBERT has
greatly advanced the field by using attention for good
handling of the linguistic features (Lan et al., 2020).
In this study used and ALBERT (A Lite BERT) model, is
an updated version of BERT and is sought after especially
for higher performance at less computational power. This
is done through the use of parameter-sharing as well as
factorized embedding parameterization came as a fancy
tool that is both efficient and productive for various NLP
tasks like text classification, sentiment analysis, and entity
recognition.
1.2 Importance of Fine-Tuning
Fine-tuning is the process of modifying a pre-trained model
to suit specific tasks or use cases in the field of machine
learning. It is now a fundamental deep learning technique,
particularly in the training of foundation models that are
used in generative AI (Lalor, 2017).
2 Methodology
This section provides the proposed methodology for
classifying text data.
1. Dataset Description: For this assignment, the
USClassActions dataset which is available at the
Hugging Face website was utilised. The
USClassActions dataset is an English dataset
comprising 3,000 complaints from the US Federal
Court. Every complaint can also be linked to a binary
outcome result where it could be a win or it could be
a loss. This job is a tough problem in text
categorisation, to promote the study on the robustness
and fairness of legal NLP. All of this data has been
annotated just using the Darrow. ai tool which makes
it highly valuable for exploring and developing
models in the field of legal text analysis and
prediction of decisions.
2. Model Selection: Due to the high quality and
accuracy of this model in textual data categorisation,
the ALBERT model was selected. ALBERT is but an
improved BERT that has lesser parameters and is
faster. It is especially good for the learning from
intricacies if legal documents found in the
USClassActions dataset because it offered very high
precision coupled with minimal memory usage and
training time.
3. Fine-Tuning Process: Very important stages during
the process of further improvement of the ALBERTbased language model are described in the article.
The text data was first preprocessed by removing any
characters that might affect the model and was then
tokenised through various operations and then
encoded for the model. The ALBERT model was
further trained using TensorFlow by Marshall et al
after being pre-trained on a colossal volume. The
accuracy measure was employed during the finetuning of the fully connected layers, which were
trained using Adam as the optimiser and Sparse
Categorical Cross-Entropy as the loss function. With
a batch size of 16, the model's parameters were
adjusted over the 30-epoch training procedure to
enhance its performance on the particular text
categorisation job.
4. Model Assessment: In this study, ROC curves,
confusion matrices, recall, accuracy, precision, and
F1 score were used to assess the ALBERT model.
The model's extensive metrics showed that it
effectively captured linguistic subtleties, and it
performed robustly by correctly categorising text
across categories.
2.1 Exploratory Data Analysis (EDA)
Figure 1: Text Data Word Cloud
Figure 1 shows the word cloud for the text data using the
USClassActions dataset. The graph clearly demonstrates
that response, user, dan, answer, never, chatgpt, and AI are
the most frequent words that are present in the dataset.
Figure 2: Pie Chart of Top 10 Most Frequent Words
Figure 2 displays a pie chart that represents the top 10 most
often occurring words in the USClassActions dataset.
According to the pie chart, “dan,” “answer,” and
“response” are three highly frequent words in the dataset.
Figure 3: Scatter Plot for Average Word Length
Average word length from the USClassActions dataset is
shown in Figure 3 as a scatter plot. The data points are
scattered rather equally, as shown by the graph, with the
majority of averages lying within the range of 5 to 8. Word
lengths naturally vary, with some points being somewhat
longer than normal (up to 12 characters).
Figure 4: Count Plot for Unique Words
Figure 4 displays a count plot that represents the frequency
of unique words in the USClassActions dataset. The graph
clearly indicates that the 0-100-word unique word count is
over 60.
Figure 5: Pie Chart for Distribution of Labels
Figure 5 displays a pie chart that represents the distribution
of labels in the USClassActions dataset. The pie chart has
two distinct colour slices: red and blue. Red represents
jailbreak, which contains 52.1% text data, and blue
represents benign, which contains 48.9% text data.
3 Experimental Result and Discussion
Figure 6: Line Graph of Accuracy and Loss for Training
and Validation
Figure 6 displays a line graph illustrating the accuracy and
loss of the ALBERT model throughout training and
validation. The graph's y-axis shows the accuracy and loss
for every epoch, while the x-axis shows the epochs, which
range from 0 to 30. The model receives a training loss of
around 0.55 and a validation loss of about 0.25, based on
the graphs. By comparison, the model achieves an
estimated 100% training accuracy and a 96% validation
accuracy.
Figure 7: ALBERT Model Classification Report
Classification report of ALBERT Model using
USClassActions dataset for text classification is presented
by figure 7. The report is comprised of two classes, namely,
benign and jailbreak. On behalf of the benign class, we
obtain the precision of 91%, the recall, equal to 100% and
the f1-score of 95% respectively. In the same way, the
model yields the jailbreak class a precision of 100%, recall
of 94% and f1-score of 97% respectively.
Table 1: ALBERT Model Performance Measures
Model Accuracy Precision Recall F1-score
ALBERT 96.00 97.00 96.00 96.00
Figure 8: Performance Analysis of ALBERT Model
The performance of the ALBERT model in text
classification using the USClassActions dataset is as
follows Figure 8 and Table 1. The bar graph in the figure
simply represents the precision of 97%, accuracy of 96%,
recall of 96%, f1-score of 96% for the ‘model’.
Figure 10: ALBERT Model Confusion Matrix
Figure 10 illustrates the confusion matrix for the ALBERT
Model for text classification of the dataset, namely the
USClassActions. The confusion matrix elaborates that for
the Albert models, true positive rate was 20, true negative
rate 31, false positive 0 and false negative rate 2.
Figure 11: ALBERT Model ROC Curve
It is possible to observe the ROC curve of the ALBERT
model in figure 11 which specifically works on the text
classification of USClassActions dataset. Within the
framework of the ROC curve model, it reaches the 98%
accuracy.
4 Conclusion
I In conclusion, it can be stated that, using the ALBERTbased language model provides high performance while
classifying the texts according to both precision, accuracy,
recall, and F1-score. Thus, the improved design of
ALBERT can be evidenced from the training process that
takes much time to complete. A vast range of assessment
criteria including the ROC curve and confusion matrix
show the effectiveness of the model in several domains.
The detailed visualisations offered in this report shed light
onto the performance of the constructed model and have
resulted in further opportunities and possibilities to
enhance the model together with real-world applications.
References
Lalor, J. (2017) ‘CIFT : Crowd-Informed Fine-Tuning to
Improve Machine Learning Ability’, Item Response
Theory (IRT) allows for measuring ability of Machine
Learning models as compared to a human population.
However, it is difficult to create a large dataset to train the
ability of deep neural network models (DNNs). We propose
Crowd-Informed F [Preprint], (November).
Lan, Z. et al. (2020) ‘ALBERT: A LITE BERT FOR
SELF-SUPERVISED LEARNING OF LANGUAGE
REPRESENTATIONS’, in 8th International Conference
on Learning Representations, ICLR 2020.
