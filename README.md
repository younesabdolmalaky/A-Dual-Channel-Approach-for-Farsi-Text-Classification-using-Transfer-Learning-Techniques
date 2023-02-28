# A-Dual-Channel-Approach-for-Farsi-Text-Classification-using-Transfer-Learning-Techniques

![218720442-7f60dee8-5a9b-47fb-bb3e-58c81aa62d47](https://user-images.githubusercontent.com/75095471/219192408-ed4dc731-dfe0-4f1c-8093-117cc9dfcded.png)
## overview
In this research, we propose a novel approach for Farsi text classification using a dual-channel model and transfer learning techniques. The model utilizes both a convolutional neural network (CNN) and a multi-layer perceptron (MLP) as channels, where the input for the MLP is a tf-idf matrix and the input for the CNN is a word embedding. To test the model, we first translated a set of Persian texts into English and then used the pre-trained English model to classify the translated texts. The results of this study demonstrate the effectiveness of this approach in classifying Farsi texts.

## dataset
In this research, two datasets were utilized to train and test the proposed dual-channel model for Farsi text classification. The first dataset, called the Amazon comments dataset, contains four million comments in total and is divided into two classes, which were used to train the model. The second dataset, called the Snapfood comments dataset, contains 80,000 comments and was used as test dataset. Both datasets are in Farsi and were translated into English before being utilized in the study. The test dataset was split into 10% for testing and 90% for re-training the model on new data. This setup allowed for a thorough evaluation of the model's performance in classifying Farsi texts by testing it on unseen data after re-training on a portion of the test dataset.

![amazon](https://user-images.githubusercontent.com/75095471/219202945-501f3253-19f4-4b4d-b717-057f02c5d9b9.png)

### text classification based on  tf-idf
TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic that is used to reflect how important a word is in a document or a collection of documents. It is commonly used in text classification and information retrieval tasks.
The TF-IDF vector is a sparse matrix that represents a document as a combination of the TF-IDF values of the words it contains. The matrix has one column for each word in the vocabulary and one row for each document. The value in the matrix at the intersection of a document and a word represents the TF-IDF value for that word in that document.
The TF-IDF value for a word is calculated as the product of its term frequency (TF) and its inverse document frequency (IDF). The TF is the number of occurrences of the word in the document, normalized by the total number of words in the document. The IDF is the logarithm of the total number of documents divided by the number of documents that contain the word[10].
The TF-IDF vector can be used as an input to a text classification algorithm. One common approach is to train a supervised machine learning model on a labeled dataset of documents and their corresponding class labels. The model is then used to predict the class label of new, unseen documents. Common machine learning algorithms used in text classification include logistic regression, support vector machines, and Naive Bayes classifiers.
### Formula: 
### TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document) 

### tfi,j =  (1)

### IDF(t) = log_e(Total number of documents / Number of documents with term t in it)

### idfi =  log(2)

### TF-IDF = TF(t) * IDF(t)


In summary, TF-IDF is a numerical statistic that is used to reflect how important a word is in a document or a collection of documents, and it is commonly used in text classification and information retrieval tasks. The TF-IDF vector is a sparse matrix that represents a document as a combination of the TF-IDF values of the words it contains. It can be used as an input to a text classification algorithm.
Tf-idf (term frequency-inverse document frequency) vectorization is not commonly used in deep learning because it can produce high-dimensional sparse data, which can be difficult for deep learning models to handle. Additionally, the model may not be able to extract meaningful features from the high-dimensional data.
To solve these problems, feature selection techniques can be used to reduce the dimensionality of the data and select the most relevant features. These techniques include techniques such as chi-squared, mutual information, and Lasso.
Once the feature selection is applied, Tf-idf can be used in deep learning by adding an embedding layer to the model, where the embedding layer will learn the dense representations of the sparse data. This embedding layer can then be followed by other layers such as convolutional or recurrent layers to extract features and classify the text data.
In summary, Tf-idf vectorization is not commonly used in deep learning due to its high dimensionality, but by using feature selection techniques, it can be used effectively in deep learning models.

![image](https://user-images.githubusercontent.com/75095471/219203216-23c6dab2-dbba-4f49-acc8-a41938020de0.png)


## text classification based on word embedding and convolutional neural networks 

One popular method for text classification is using convolutional neural networks (CNNs) with word embeddings.Word embeddings are a way of representing words as dense vectors in a high-dimensional space, where each dimension represents a semantic or syntactic feature of the word. These embeddings are typically pre-trained on a large corpus of text using techniques such as word2vec or GloVe.In a CNN-based text classification model, the input to the network is a set of word embeddings for the words in the text. These embeddings are passed through multiple layers of convolution and pooling, which extract features from the input and reduce its dimensionality. The output of the CNN is then passed through one or more fully connected layers, which use the extracted features to make a final prediction about the text's category or sentiment.One key advantage of using CNNs with word embeddings for text classification is that they are able to handle variable-length input text, which is common in many NLP tasks. This is because the convolution and pooling layers are able to extract features from the input regardless of its length, and the fully connected layers can then use these features to make a prediction.Another advantage of using word embeddings in a CNN-based text classification model is that they capture the semantic and syntactic relationships between words, which is important for understanding the meaning of a text. For example, the embeddings for words like "happy" and "joyful" will be similar, which allows the model to recognize that they are related in meaning.In addition, CNNs are able to handle a large amount of parameters which results in a better performance compared to other architectures such as RNNs, LSTMs and GRUs. Overall, using CNNs with word embeddings is a powerful method for text classification that can handle variable-length input text and capture the semantic and syntactic relationships between words.

![image](https://user-images.githubusercontent.com/75095471/219203330-10f02e0c-f2ba-4a7e-9daa-5b432cb1025e.png)


## text classification based on  dual-channel deep learning
Dual-channel deep learning is a method of text classification that utilizes two channels of information to make predictions. In this approach, two different neural networks are used to extract features from the input text, one for the character-level representation and the other for the word-level representation
 .One popular method for implementing dual-channel deep learning is to use a convolutional neural network (CNN) for the character-level representation and a recurrent neural network (RNN) for the word-level representation. 

A convolutional neural network (CNN) is used for the character-level representation. CNNs are typically used for image classification tasks, but they can also be applied to text data. In this case, a CNN is used to extract features from the input text by analyzing the individual characters and their combinations. This can be useful for capturing the structure and meaning of words that may be difficult for a traditional model to understand, such as misspellings or slang.
A recurrent neural network (RNN) is used for the word-level representation. RNNs are a type of neural network that are particularly well-suited to handling sequential data, such as text. In this case, an RNN is used to analyze the input text at the word level, taking into account the context and relationships between words.
The outputs of these two networks are then combined and used to make the final prediction. This approach has been shown to be effective for a variety of text classification tasks, including sentiment analysis, topic classification, and named entity recognition. In particular, it has been found to be particularly useful for handling informal text, such as social media posts, which can be difficult for traditional models to understand.
This approach of dual-channel deep learning allows for more accurate predictions by combining the information from two channels, character-level and word-level representation. By utilizing CNN and RNN networks, it can extract features from the text data that traditional models may miss, making it more effective in handling informal text.

## text classification based on  dual-channel deep learning (combine tf-idf and cnn)
TF-IDF (term frequency-inverse document frequency) is a method used to quantify the importance of words in a document. It takes into account both the frequency of a word within a document, as well as how often the word appears in the entire corpus of documents. The main advantage of TF-IDF is that it can effectively identify and assign importance to rare, but relevant words that may not be captured by other methods that only consider word frequency.
Word embeddings, on the other hand, are a way of representing words as numerical vectors. The advantage of word embeddings is that they can capture the meaning and context of words in a way that traditional one-hot encodings cannot. By learning the relationships between words, embeddings can also be used to perform tasks like language translation and text generation.
By using a two-channel deep learning architecture that combines the strengths of TF-IDF and word embeddings, you can take advantage of the complementary information provided by each method. The TF-IDF channel can capture the importance of rare words, while the word embeddings channel can capture the meaning and context of words. By having both channels, the model can reach a better consensus about the importance of words and this can improve the performance of the model.
Overall, the combination of these two methods can provide a more comprehensive understanding of the text and lead to better performance on natural language processing tasks.

## machine translation
One approach to addressing the challenges of Farsi text classification is to use machine translation to translate the text into a more widely-used language, such as English. By doing so, one can leverage pre-existing models and resources developed for the target language to classify the text.One benefit of this approach is that it allows for greater access to labeled data, as there is likely more available in the target language than in the source language. Additionally, pre-existing models for the target language may be more accurate and perform better on the translated text than models developed specifically for the source language.
Another benefit is that the translation process can help to reduce the complexity of the text, as machine translation can simplify the grammar and vocabulary of the text. This can make it easier for models to classify the text accurately.
It's important to note that, this approach may not be suitable for all types of text classification task, as it might lose some of the cultural, contextual, and idiomatic nuances of the source language. Moreover, the translation quality is a crucial factor to consider, and not all the machine translation are created equal, so it's recommended to evaluate different translation engines before using it[7].
In conclusion, using machine translation to translate Farsi text into English can be an effective approach for addressing the challenges of Farsi text classification. By leveraging pre-existing models and resources for the target language, this approach can improve the accuracy and performance of text classification models. However, it's important to keep in mind the potential limitations and evaluate the translation quality before using it.

![illustration_deck-automate](https://user-images.githubusercontent.com/75095471/219204101-3703986c-4051-4f71-9749-f3f5526ca7f2.png)

## Farsi Text Classification Based on  dual-chanal and transfer Learning
In this study, we proposed a method for classifying Persian texts using an English language model. The approach involves translating the Persian texts into English and then using a pre-trained English text classification model to classify the translated text. The model we used for classifying English texts is a two-channel model that consists of a CNN and an MLP.
The input of the MLP is the tf-idf matrix, which is a common representation of text used in natural language processing tasks. To reduce the computational complexity, we used feature selection methods to select the top 1000 features from the vector. This allowed us to focus on the most informative features while keeping the model computationally efficient[8].
The input of the CNN is the word embeddings, which are dense representations of words that capture the semantic and syntactic information of the text[9]. By using word embeddings, the CNN can learn the underlying patterns of the text, and it can also handle out-of-vocabulary words.
The advantages of this method are multiple, firstly, by translating Persian texts into English, the model can leverage pre-existing models and resources developed for the target language, which can improve the accuracy and performance of text classification. Additionally, the translation process can help to reduce the complexity of the text, as machine translation can simplify the grammar and vocabulary of the text. This can make it easier for models to classify the text accurately.
Another advantage is that by using a two-channel model, we can take advantage of the strengths of both CNNs and MLPs. CNNs are good at learning patterns in the text, while MLPs are good at handling structured data. Combining these two models allows us to exploit the strengths of both models and achieve better performance than using a single model.
Furthermore, by using feature selection methods[8], we reduced the dimensionality of the data and eliminated noisy features that might have led to overfitting and poor generalization. Additionally, by using word embeddings as input, we can learn the underlying patterns of the text, which can help improve the performance of the model .
In conclusion, our proposed method of using an English language model to classify Persian texts after translating them, is an effective approach that addresses the challenges of Farsi text classification. By leveraging pre-existing models and resources for the target language, this approach can improve the accuracy and performance of text classification models, and by using a two-channel model, feature selection, and word embeddings, we can further improve the performance.


## results on amazon dataset
<table>
  <tr>
    <th>Model</th>
    <th>TF-IDF</th>
    <th>CNN</th>
    <th>Multi-channel</th>
  </tr>
  <tr>
    <td>Accuracy - Train</td>
    <td>0.90</td>
    <td>0.95</td>
    <td>0.955</td>
  </tr>
  <tr>
    <td>Accuracy - Test</td>
    <td>0.90</td>
    <td>0.94.91</td>
    <td>0.9512</td>
  </tr>
</table>

The results of the dual-channel deep learning model applied to an English dataset show that this approach is an effective method for text classification. The model utilizes both a convolutional neural network (CNN) and a term frequency-inverse document frequency (TF-IDF) matrix as inputs. The accuracy of the model on the training set was 0.955, while the accuracy on the test set was 0.9512 These results indicate that the model has a high level of performance and generalizes well to new data. Additionally, the results of this model surpassed the single channel models (CNN with accuracy of 0.9491 and TF-IDF with accuracy of 0.90) which suggests that the dual-channel approach of combining CNN and TF-IDF inputs offers a significant improvement in performance. Overall, the use of this dual-channel model for text classification is a promising method that could be applied to other datasets and natural language processing tasks.

## Results on snappfood dataset
In the two-channel model (without using transfer learning), we have trained the model on Persian data

<table>
  <tr>
    <th>Model</th>
    <th>Multi-channel-TL</th>
    <th>Multi-channel</th>
  </tr>
  <tr>
    <td>Accuracy - Train</td>
    <td>0.85</td>
    <td>0.70</td>
  </tr>
  <tr>
    <td>Accuracy - Test</td>
    <td>0.84</td>
    <td>0.69</td>
  </tr>
</table>

The results of the dual-channel deep learning model with transfer learning applied to a dataset show that this approach is an effective method for text classification. The model utilizes both a convolutional neural network (CNN) and a term frequency-inverse document frequency (TF-IDF) matrix as inputs. The accuracy of the transfer learning model on the training set was 0.90 and on the test set was 0.85. These results indicate that the model has a high level of performance and generalizes well to new data. Additionally, the results of this model surpassed the dual-channel models (without transfer learning) with accuracy of 0.70 and 0.69 respectively on training and test set, which suggests that using transfer learning with this dual-channel approach of combining CNN and TF-IDF inputs offers a significant improvement in performance. Overall, the use of this dual-channel model with transfer learning for text classification is a promising method that could be applied to other datasets and natural language processing tasks.
