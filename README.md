
![MachineLearning](https://github.com/SumitxThokar/MachineLearning/blob/main/images/pexels-tara-winstead-8386434.jpg)
# **Machine Learning**
| **Books and Resources** | **Status of Completion** |
| ----- | -----|
| 1. **Machine Learning Crash Course** | 🟢 |
| 2. **Machine Learning From Scratch** | 🔴 |
| 3. **A Comprehensive Guide to Machine Learning** | 🔴 |
| 4. **Hands On Machine Learning with Scikit Learn, Keras and TensorFlow** | 🟡 |


| **Projects** | **Status of Completion** |
| ----- | -----|
| 1. [**California House Price Prediction**](https://github.com/SumitxThokar/California-House-Price-Prediction) | :white_check_mark: |
| 2. [**Exploratory Data Analysis on Dataset-Terrorism**](https://github.com/SumitxThokar/LetsGrowMoreProjects/blob/main/Global%20terrorism/GlobalTerrrorism.ipynb) | :white_check_mark: |
| 3. [**Image to Pencil Sketch with Python**](https://github.com/SumitxThokar/LetsGrowMoreProjects/blob/main/Pencil%20image%20converter/Pencil_Sketch_Converter.ipynb)  |:white_check_mark: |
| 4. [**Iris Flowers Classification ML Project**](https://github.com/SumitxThokar/LetsGrowMoreProjects/blob/main/Iris/IrisFlowerClassificationwithKNN.ipynb) | :white_check_mark: | 
| 5. [**Titanic-Machine Learning From Disaster**](https://github.com/SumitxThokar/Titanic---Machine-Learning-from-Disaster/blob/main/Titanic%20prediction/titanic.ipynb) | :white_check_mark: | 

## Day 1 of #66DaysofML
- **Course**: **Machine Learning Crash Course** <br>
:large_blue_diamond: I am starting my machine learning with **Google Machine Learning Crash Course**.<br>
Today I learned what machine learning is, common terminology, and common components involved and types of machine learning.<br>
### Machine Learning
Machine learning is part of the broader field of artificial intelligence. This field is concerned with the capability of machines to perform activities using human-like intelligence without being explictly programmed.<br>
### Types of Machine Learning
Within machine learning there are several different kinds of tasks or techniques:<br>
**Supervised learning**: Every training sample from the dataset has a corresponding label or output value associated with it. As a result, the algorithm learns to predict labels or output values.<br>
**Unsupervised learning**: There are no labels for the training data. A machine learning algorithm tries to learn the underlying patterns or distributions that govern the data.<br>
**Reinforcement learning**: The algorithm figures out which actions to take in a situation to maximize a reward (in the form of a number) on the way to reaching a specific goal. This is a completely different approach than supervised and unsupervised learning.<br>

### Regression vs. classification<br>
A **regression model** predicts continuous values. For example, regression models make predictions that answer questions like the following:<br>
- What is the value of a house in California?<br>-What is the probability that a user will click on this ad?
<br>A **classification model** predicts discrete values. For example, classification models make predictions that answer questions like the following:<br>
- Is a given email message spam or not spam?<br>
- Is this an image of a dog, a cat, or a hamster?<br>

## Day 2 of #66DaysofML.
- **Course**: **Machine Learning Crash Course** <br>
:large_blue_diamond: Onto the journey of learning ML, Today I relate weights and biases in machine learning to slope and offset in line fitting, understand "loss" in general and squared loss in particular.<br>
### Training a model
Training a model simply means fitting labeled data(examples) into models from where the model learn good values for all weights and bias. A machine learning algorithm builds a model by observing many examples and determinines best parameters for the dataset that reduces **loss** or error in prediction. <br>
### Loss
Loss or Error is the number representing how bad the prediction was on a example  i.e. Loss= (Observation - Prediction).
![Train and Loss](https://github.com/SumitxThokar/MachineLearning/blob/main/images/LossSideBySide.png)
In the image above, The blue line represent predictions made by the model, the yellow bubble represents actual value and the red line represents **loss**.
<br> 
### Loss function.
The method of measuring loss in meaning fashion is loss function. The most popular loss function is Mean Squared Error (MSE).<br>
**Mean Squared Error** is the average squared loss of every example in the dataset.
![MSE](https://github.com/SumitxThokar/MachineLearning/blob/main/images/mse.png)
<br>
## Day 3 of #66DaysofML.
- **Course**: **Machine Learning Crash Course** <br>
:large_blue_diamond: On my journey of learning Learning ML. These are my finding and learnings today:
### Reducing Loss<br>
Achieving optimal model performance by iteratively adjusting parameters to reduce loss. A Machine Learning model is trained by starting with an initial guess for the weights and bias and iteratively adjusting those guesses until learning the weights and bias with the lowest possible loss.
![Reducing Loss](https://github.com/SumitxThokar/MachineLearning/blob/main/images/Iterative.png)
### Gradient Descent
Gradient descent is an optimization algorithm used to minimize a function by iteratively adjusting the parameters in the direction of the negative gradient of the function. It is a widely used algorithm in machine learning to find the best parameters of the model that minimize the loss. The gradient always points in the direction of steepest increase in the loss function. The gradient descent algorithm takes a step in the direction of the negative gradient in order to reduce loss as quickly as possible.
![Gradient Descent](https://github.com/SumitxThokar/MachineLearning/blob/main/images/gd.png)

## Day 4 of #66DaysofML.
- **Course**: **Machine Learning Crash Course** <br>
### Learning Rate
:large_blue_diamond: The Gradient Descent algorithm is a powerful tool for optimizing machine learning models by adjusting the parameters of the model in order to minimize the error. One of the key components of this algorithm is the **learning rate**, which is a scalar value that is multiplied by the gradient in order to determine the next point in the optimization process. For example, if the gradient magnitude is 2.5 and the learning rate is 0.01, the algorithm will move to a point that is 0.025 units away from the previous point. This small step size allows for a gradual and controlled optimization process. In addition, gradient descent can also be applied to feature sets that contain multiple features, making it a versatile algorithm for a wide range of machine learning tasks. 
<br> **When the Learning rate is small.**
![Small Learning rate](https://github.com/SumitxThokar/MachineLearning/blob/main/images/rate.png)
<br>**What Learning rate should be like.**
![Good Learning rate](https://github.com/SumitxThokar/MachineLearning/blob/main/images/goodrate.png)
<br>A variant of this algorithm is **Stochastic Gradient Descent** which uses random samples from the data to update the parameters.

## Day 5 of #66DaysofML.
- **Course**: **Machine Learning Crash Course** <br>
:large_blue_diamond: Today, I was introduced to TensorFlow, an open-source platform for machine learning. TensorFlow's APIs are organized in a hierarchical structure, with high-level APIs built on top of low-level APIs. As a beginner, I will be using the high-level APIs, specifically Keras, which simplifies the process of building and training machine learning models. I revisited my knowledge of NumPy and pandas to better understand the code written in tf.keras. I also became familiar with the use of Google Colab, a tool that allows for easy collaboration and code execution. One of the main objectives of the day was to learn how to use linear regression code in tf.keras. I also evaluated loss curves and practiced tuning hyperparameters to improve the performance of my model. Overall, it was a valuable experience and I look forward to continuing my learning journey with TensorFlow.
![Summary](https://github.com/SumitxThokar/MachineLearning/blob/main/images/summarytuninghyperparameter.jpg)

## Day 6 of #66DaysofML.
- **Course**: **Machine Learning Crash Course** <br>
:large_blue_diamond: Today, I got to develop intuition about overfitting and got idea on how to determine whether a model is good or not.<br>
In machine learning, it is important to consider the ability of a model to generalize to new, unseen data. This is known as generalization. On the other hand, overfitting occurs when a model is too complex and has learned noise from the training data, rather than the underlying pattern. An overfit model will have a low loss during training, but will perform poorly on new data.<br>
So, how can we determine whether a model is good or not? One way is to evaluate its performance on new data. To do this, we can divide our dataset into a training set and a test set. The training set is used to fit the model, while the test set is used to evaluate the model's performance. If the model performs well on the test set, it is an indicator that the model is good.<br>
In summary, generalization and overfitting are important concepts in machine learning. To determine whether a model is good, we can evaluate its performance on new data by using a train and test set. A good performance on the test set is an indicator of a good model.
<br>

## Day 7 of #66DaysofML.
- **Course**: **Machine Learning Crash Course** <br>
:large_blue_diamond: When it comes to building a machine learning model, one of the most important steps is dividing your data set into a training set and a test set. This is because having separate data sets for training and testing allows you to gauge the performance of your model in a more accurate and reliable way.
![TrainingTest data](https://github.com/SumitxThokar/MachineLearning/blob/main/images/Trainingtest.jpg)
<br>
The training set is used to train your model, and the test set is used to evaluate the performance of the model once it has been trained. The larger the training set, the better your model will learn and the more accurate it will be. This is because a larger training set provides the model with more data to learn from and generalize from.<br>
On the other hand, the larger the test set, the more confident you can be in the performance of your model. This is because a larger test set provides a more representative sample of the data and can give you a better idea of how well your model will perform on unseen data.
<br>
However, it is important to keep in mind that when working with smaller data sets, you may need to use cross-validation techniques to effectively divide your data into training and test sets. Cross-validation is a statistical method that allows you to train and test your model on different subsets of your data, giving you a more robust estimate of the model's performance.
<br>
It is also important to note that it is never a good idea to train your model on the test data. This is because the test data is meant to be used as a way to evaluate the model's performance on unseen data, and training on this data would lead to an overestimation of the model's performance.
<br>


## Day 8 of #66DaysofML.
- **Course**: **Machine Learning Crash Course** <br>
:large_blue_diamond:**Validation set**<br>
Dividing your data into training, validation, and test sets is a common practice in machine learning to ensure that the model is properly validated and generalizes well to new data. The first step in this workflow is to train the model on the training set, which allows it to learn the patterns and relationships in the data.<br>
![Validation set](https://github.com/SumitxThokar/MachineLearning/blob/main/images/validation_set.jpg)
The next step is to evaluate the model on the validation set, which is a set of data that the model has not seen before. This allows you to see how well the model is performing on new data and to adjust the hyperparameters, such as the learning rate or the number of hidden layers, accordingly. Once you have found the best performing model on the validation set, you can then confirm its performance on the test set, which is a final set of unseen data. This workflow greatly reduces the chance of overfitting, which occurs when a model is overly optimized for the training data and performs poorly on new data.<br>
![Workflow](https://github.com/SumitxThokar/MachineLearning/blob/main/images/better_workflow.jpg)
<br>

## Day 9 of #66DaysofML.
- **Course**: **Machine Learning Crash Course** <br>
:large_blue_diamond: **Feature Engineering**
<br>In real-life scenarios, data is not always presented in a neat, organized format. Often, data comes in the form of record or protocol buffers, requiring the extraction of useful information and the creation of features through a process known as feature engineering. Unlike traditional programming, machine learning (ML) focuses on representation, and improving the quality of the features can significantly increase the performance of the model. It is estimated that 70% of an ML engineer's time is spent on this process.<br>
![FeatureEngineering](https://github.com/SumitxThokar/MachineLearning/blob/main/images/FeatureEng.jpg)
Properties of good features include having clear and obvious meanings, non-zero values more than a small handful of times, not taking on magic values, not changing over time, and not having extreme outliers. In order to clean and prepare the data for machine learning, it is important to scale the feature values, handle extreme outliers, and remove or replace duplicate values, bad feature values, bad labels, and null values. The key to successful feature engineering is understanding your data, visualizing it, and transforming integer values to floating points and categorical features (strings) into numeric values.
<br>
In summary, feature engineering is a crucial aspect of machine learning that requires a thorough understanding of the data, attention to detail, and a willingness to experiment with different approaches. By following best practices for data cleaning, visualization, and mapping, you can create high-quality features that will improve the performance of your machine learning model.

## Day 10 of #66DaysofML.
- **Course**: **Machine Learning Crash Course** <br>
:large_blue_diamond: **Feature Cross** <br>
Creating synthetic features through feature crosses can be a powerful strategy for improving the predictive abilities of machine learning models, especially when working with large data sets. <br>
One way to implement feature crosses is through the use of **tf.feature_column** methods in TensorFlow. By representing features in different ways, such as through the use of bins, it is possible to create new, synthetic features by crossing these bins.
<br>
This technique can be especially useful for linear learners, which scale well to large data sets and can benefit from the added complexity provided by feature crosses. Additionally, this technique can be used in conjunction with neural networks for an even more powerful approach to learning highly complex models.
<br>
Overall, the use of feature crosses can be a valuable tool for any data scientist looking to improve the performance of their machine learning models.

## Day 11 of #66DaysofML.
- **Course**: **Machine Learning Crash Course** <br>
:large_blue_diamond: **Regularization: Simplifying Your Model to Improve Generalization**

When training a machine learning model, it's important to not only minimize the loss on the training data, but also ensure that the model generalizes well to new, unseen data. One way to achieve this is through regularization, a technique that aims to penalize more complex models in order to prevent overfitting.
<br>
One common form of regularization is L2 regularization, which adds a term to the loss function that is the sum of the squares of all the feature weights. This has the effect of "shrinking" the weights towards zero, effectively simplifying the model.
<br>
To see the effects of regularization, we can plot a learning curve, which shows how the model's performance on the training and validation data changes as we increase the number of iterations during training. In the case of overfitting, we will see the training loss decreasing while the validation loss increases. By adding regularization, we can observe that both the training and validation loss decrease, indicating that the model is not only fitting the training data well, but also generalizing well to new data.
<br>
In summary, regularization is a powerful technique that can help improve the generalization of your machine learning model by simplifying it and preventing overfitting. L2 regularization is a popular form of regularization that involves adding a term to the loss function that is the sum of the squares of the feature weights. By observing the learning curve, we can see the effects of regularization and choose the right regularization term to achieve good generalization and avoiding overfitting.
<br>
## Day 12 of #66DaysofML.
**Course**: **Machine Learning Crash Course** <br>
:large_blue_diamond: On my journey of Learning Machine Learning, The topics I covered today are Logistic Regression, Regularization in Logistic Regression, Logistic Regression for Classification, Evaluation Metrics like Accuracy, Precision and Recall, F1-Score, ROC Curve and AUC, True vs False: Positive vs Negative.<br>
**Summary** <br>
Logistic Regression is a method for predicting the probability of a classification problem, with outputs between 0 and 1 determined by a sigmoid function. Regularization can be used to prevent overfitting by penalizing large weights (L2) or limiting the training steps/learning rate (early stopping). Evaluation metrics for classification include accuracy, precision (correct positive predictions), and recall (correctly identified positive cases). ROC curve and AUC are also used to evaluate model performance. Prediction bias can be analyzed using calibration plots. The F1-score is a harmonic mean of precision and recall.

## Day 13 of #66DaysofML.
**Course**: **Machine Learning Crash Course** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I covered topic Regularization for sparsity, L1 Regularization, L1 vs L2 Regularization. 
### Regularization for sparsity.
Regularization is a technique used in machine learning to prevent overfitting by adding a penalty term to the cost function. One form of regularization is sparsity regularization, which aims to produce sparse models with a small number of non-zero parameters. This can be useful in situations where the number of features is large, such as in image or text data. However, sparsity regularization can also lead to issues such as a large model size and noise coefficients.
<br>
L1 and L2 regularization are two common forms of regularization used in machine learning. L1 regularization adds a penalty term to the cost function that is proportional to the absolute value of the parameters, while L2 regularization adds a penalty term proportional to the square of the parameters. L1 regularization tends to produce sparse models, as it tends to drive some of the parameters to zero. On the other hand, L2 regularization tends to produce models where all the parameters are small, but non-zero. In general, L1 regularization is more likely to produce sparse models, while L2 regularization is less likely to produce sparse models, but is less likely to produce overfitting. <br>
![l1vl2](https://github.com/SumitxThokar/MachineLearning/blob/main/images/l1vl2.jpg)

## Day 14 of #66DaysofML.
**Course**: **Machine Learning Crash Course** <br>
:large_blue_diamond: On my journey of learning Machine Learning, I developed intuition about neural networks particularly about hidden layers and activation functions.
<br>
A neural network is a type of machine learning algorithm modeled after the structure and function of the human brain. It is composed of layers of interconnected "neurons," which process and transmit information. The layers between the input and output layers are called "hidden layers." These layers allow the network to learn and represent more complex and abstract patterns in the data.
<br>
Activation functions are used in the neurons of a neural network to determine the output of that neuron given a set of inputs. They introduce non-linearity into the network, allowing it to learn and represent a wider range of patterns. Common activation functions include sigmoid, ReLU, and tanh.<br>
### Playground Exercise on Neural net (hidden layers and activation functions).
![Neural Nets ](https://github.com/SumitxThokar/MachineLearning/blob/main/images/neural.jpg)

## Day 15 of #66DaysofML.
**Course**: **Machine Learning Crash Course** <br>
:large_blue_diamond: On my journey of Learning Machine Learning, I developed some intuition around backpropagation, backpropagation's failure cases and the most common way to regularize a neural network.<br>
**Backpropagation** is a widely used algorithm for training neural networks. It allows for the efficient calculation of the gradients of the loss function with respect to the weights of the network, which are used to update the weights in order to minimize the loss.
<br>
One common failure case of backpropagation is the vanishing gradient problem, where the gradients become extremely small and the network is unable to learn effectively. Another failure case is the exploding gradient problem, where the gradients become extremely large and cause the network to diverge.
<br>
A solution for the Dead ReLU units is Leaky ReLU activation function, which allows a small non-zero gradient when the input is negative.
<br>
**Dropout** is a regularization technique used in training neural networks. It randomly sets a fraction of the input units to zero during training, which helps to prevent overfitting by reducing the dependence of the output on any specific input unit.

## Day 16 of #66DaysofML.
**Course**: **Machine Learning Crash Course** <br>
:large_blue_diamond: On my journey of Learning Machine Learning, I developed an understanding of multi-class classification problems, particularly Softmax.
### Multi-Class Neural Networks.
A multiclass neural network is a type of machine learning model that is used to solve problems with multiple classes. It is trained to recognize and classify multiple different types of inputs, rather than just two classes as in binary classification.<br>
**Multi-Class Neural Nets method** <br>
**One Vs All Multi-class** <br>
It is a method for solving multiclass problems by training a separate binary classifier for each class, where the class is classified as 1 and all other classes are classified as 0. This method is efficient with a small number of classes, but becomes less efficient as the number of classes increases.

![Onevall](https://github.com/SumitxThokar/MachineLearning/blob/main/images/onevall.jpg)
<br>
**Softmax** <br>
Softmax is a method for assigning decimal probabilities to each class in a multiclass problem, where the probabilities must add up to 1.0. This constraint helps the training converge more quickly. There are two options for implementing Softmax: Full Softmax, which calculates a probability for every possible class, and Candidate Sampling, which only calculates probabilities for positive labels. Full Softmax is relatively costly in terms of time and memory, while Candidate Sampling is more efficient.


![Softmax](https://github.com/SumitxThokar/MachineLearning/blob/main/images/softmax.jpg)


## Day 17 of #66DaysofML.
**Course**: **Machine Learning Crash Course** <br>
:large_blue_diamond: On my journey of Learning Machine Learning, Today I developed Softmax solutions in TensorFlow.<br>
To use Softmax in TensorFlow to classify handwritten digits, you can follow these steps:
- Import TensorFlow and other necessary libraries.
- Load the dataset of handwritten digits (such as MNIST) and preprocess it.
- Define the model architecture, including the input layer, hidden layers, and output layer with a Softmax activation function.
- Compile the model by specifying the optimizer, loss function, and metrics.
- Train the model on the dataset.
- Evaluate the model on a test set of images and use the Softmax output to predict the class of each image.


![part 1](https://github.com/SumitxThokar/MachineLearning/blob/main/images/1.jpg)
![part 2](https://github.com/SumitxThokar/MachineLearning/blob/main/images/2.jpg)
![part 3](https://github.com/SumitxThokar/MachineLearning/blob/main/images/3.jpg)
### Output
![Accuracy](https://github.com/SumitxThokar/MachineLearning/blob/main/images/download.png)
<br>
**It shows that the model has accuracy of 98%.**

![OUtput](https://github.com/SumitxThokar/MachineLearning/blob/main/images/downloa2.png)


## Day 18 of #66DaysofML.
**Course**: **Machine Learning Crash Course** <br>
:large_blue_diamond: On my journey of Learning Machine Learning, I learned what an embedding is and what it's for, how embeddings encode semantic relation, how to use embeddings and how to train meaningful embeddings (using word2vec, for example).

### **Embeddings**
In machine learning, an embedding is a way of representing data in a lower-dimensional space. This can be useful for tasks such as natural language processing and computer vision. Common techniques for creating embeddings include:<br>
1. Word embeddings: These are used to represent words in a way that captures their meaning. One popular method for creating word embeddings is the word2vec algorithm.<br>
2. Sentence embeddings: These are used to represent entire sentences or paragraphs. One popular method for creating sentence embeddings is the Universal Sentence Encoder.<br>
3. Image embeddings: These are used to represent images in a way that captures their content. One popular method for creating image embeddings is the convolutional neural network (CNN).<br>
4. Graph Embedding: These are used to represent graph data in a way that captures the structure of the graph. One popular method for creating graph embeddings is Graph Attention Network (GAT).<br>
5. Audio Embedding: These are used to represent audio data in a way that captures the features of the audio. One popular method for creating audio embeddings is the Mel-Frequency Cepstral Coefficients (MFCCs)
<br>
Embeddings can be used as inputs to other machine learning models, such as neural networks, to improve their performance on tasks such as text classification and image recognition.


![Recommendation System](https://github.com/SumitxThokar/MachineLearning/blob/main/images/recommend.jpg)


## Day 19 of #66DaysofML.
**Course**: **Machine Learning Crash Course** <br>
:large_blue_diamond: On my journey of Learning Machine Learning, Today I understood the breadth of components in a production ML system, pros and cons of static and dynamic training and inference, estimated training and serving needs for real-word scenarios and understand data dependencies in production ML system.
### **Summary**
Machine learning systems are programs that can learn from data and make predictions without explicit programming. There are two types of training: static and dynamic. Static training uses a fixed dataset, while dynamic training updates the model as new data becomes available. Inference, the process of using a trained model to make predictions, can also be static or dynamic. Data dependencies refer to the relationship between input and output data, and how changes in one affect the other.

## Day 20 of #66DaysofML.
**Course**: **Machine Learning Crash Course** <br>
:large_blue_diamond: On my journey of Learning Machine Learning, I got aware of common human biases that can inadvertently be reproduced by ML algorithms, Fairness, Types of bias, ways to identify bias and evaluating for bias.
### Summary
Fairness in ML refers to unbiased and impartial predictions from machine learning models regardless of personal characteristics. Types of bias include sampling, confounding, and predictive bias. Identifying bias can be done by analyzing data and examining feature engineering, and evaluating bias involves measuring the impact of predictions on different groups.

## Day 21 of #66DaysofML.
**Course**: **Machine Learning Crash Course** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I determined flawa in real-world ML model. The course discussed on two examples, one for Cancer Prediction where a model was used to predict the probability of a patient having cancer based on medical records and another for 18th Century Literature where a model was used to predict the political affiliation of authors based on the metaphorical language used. With this I have completed the ML crash course by Google. I will be continuing my ML journey with **Hands on Machine Learning  with scikit Learn, Keras & Tensorflow**. I am excited to learn more and looking forward for tomorrow.

## Day 22 of #66DaysofML.
**Course**: **Hands on Machine Learning with scikit Learn, Keras & Tensorflow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I continued my ML journey with Hand on ML. After a detailed look at the Preface, I jumped to End-to-End Machine Learning Project. The project is California House Prediction. I fetched the data with function, load the data using pandas, summarize the data, visualized the histogram for each numerical attribute, created test and train set using train_test_split from sklearn.model_selection, visualized the geographical data using scatterplot.


![](https://github.com/SumitxThokar/Machine_Learning_Journey/blob/main/images/scatter.jpg)  


## Day 23 of #66DaysofML.
**Course**: **Hands on Machine Learning with scikit Learn, Keras & Tensorflow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I continued working on End-to-End Machine Learning Project (California House Prediction). After splitting the dataset and I visualized a scatter matrix of their correlation where I got to know the best attributes correlated. I exprerimented with attribute combinations, prepared the data for Machine Learning Algorithms by cleaning the data, Encoding the categorial data. I also got to learn more on Feature Scaling, Transformation Pipelines. **ColumnTransformer** was new to me and which I found pretty cool.  

## Day 24 of #66DaysofML.
**Course**: **Hands on Machine Learning with scikit Learn, Keras & Tensorflow** <br>
:large_blue_diamond: On my journey of learning Machine Learning,With meticulous precision, I fine-tuned my model by employing both Grid Search and Randomized Search. When the hyperparameter search space is extensive, it's advisable to opt for the use of RandomizedSearchCV for optimal results. Today I worked on the same machine learning project. I trained a Linear Regression model on my data and checked its accuracy. Unfortunately, the model didn't perform well and was underfitting the data. I then tried a more powerful model called DecisionTreeRegressor, but the results showed overfitting. To overcome this, I applied cross-validation to the model.

## Day 25 of #66DaysofML.
**Course**: **Hands on Machine Learning with scikit Learn, Keras & Tensorflow** <br>
:large_blue_diamond: On my journey of learning Machine Learning,With meticulous precision, I fine-tuned my model by employing both Grid Search and Randomized Search. When the hyperparameter search space is extensive, it's advisable to opt for the use of RandomizedSearchCV for optimal results.

## Day 26 of #66DaysofML.
**Course**: **Hands on Machine Learning with scikit Learn, Keras & Tensorflow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, After completing the previous Regression project of predicting housing price, I moved towards Classification. Classification is all about predictiong classes.The dataset used was MNIST and it was loaded using SciKit-Learn libraries. The data was then divided into two parts - training data and test data. A type of classifier called "SGD classifier" was trained to differentiate between two classes, whether the data was 5 or not.

## Day 27 of #66DaysofML.
**Course**: **Hands on Machine Learning with scikit Learn, Keras & Tensorflow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I dived more into the world of performance mearures in binary classification problems. I had the chance to work with cross-validation in classification problems. I also learned about the confusion matrix and put it into practice. I was able to implemet accuracy metrics such as precision and recall and F1-score to gain a deeper understanding of the meodel's preformance.


## Day 28 of #66DaysofML.
**Course**: **Hands on Machine Learning with scikit Learn, Keras & Tensorflow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I gained insights into the precision/recall tradeoff in machine learning. I learned that scikit-learn does not allow direct threshold setting, but it does provide access to the decision scores used for predictions. This is done by using the classifier's predict() method and its decision_function() method. To compute precision and recall, I used the cross_val_predict and precision_recall_curve functions.


## Day 29 of #66DaysofML.
**Course**: **Hands on Machine Learning with scikit Learn, Keras & Tensorflow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I delved into the world of precision and recall tradeoff and discovered the importance of selecting a good balance between these two metrics. I created a precision against recall graph to visually represent this tradeoff. Additionally, I learned about the Receiver Operating Characteristic (ROC) curve and used it to plot the performance of a Random Forest Classifier. The results were impressive with a precision score of 99% and a recall score of 87%. These graphs are useful tools in understanding the tradeoff between precision and recall and can be used to help improve the performance of machine learning models.

## Day 30 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I learned about Multiclass Classification (which can distinguish between more than two classes), One Vs All and One Vs One strategy to train a binary classifier. Then I worked on error analysis which is a way to improve the model by analyzing the types of error the model makes.

## Day 31 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I gained insights on Multilabel Classification (where each instance have multiple classes) with example. In the example we create a KNeighborsClassifier instance as it supports multilabel classification. I dived further into the types of Classification and learned about Multioutput Classification.

## Day 32 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I continued my ML journey with Chapter-4 ( Training Models). I stated by looking at the Linear regression Model, one of the simplest models. I observed how the Linear Regression works closely in both linear and vectorized form. Training a model means settinf its parameters so that model best fits the training set. To measure how well or poorly the dataset are fitted, I looked at common performance measure of Regression Model i.e RMS and its equation. I got insights on the Normal Equation which is used to find the value of parameter (bias) that minimizes the cost function.
![](https://github.com/SumitxThokar/Machine-Learning-Journey/blob/main/images/image1.jpg)

## Day 33 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I got to revise my learnings on Gradient Descent. The main idea of Gradient Descent is to tweak parameters iteratively in order to minimize a cost function. I dived more on Gradient Descent learning how feature scaling helps GD, cost function of Linear Regression model being always convec i.e when we pick two points in the curve of the function, the two points in the curve of the function, the line segment joining them won't cut the curve meaning it has no local minimum. And lastly I implemented the concepts of GD in to lab in jupyter-notebook.

## Day 34 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I delved more into Gradient Descent. I successfully implemented Linear Regression using an iterative optimization approach called Stochastic Gradient Descent.  
Key Points:
1. Stochastic Gradient Descent (SGD) is a type of optimization algorithm used in machine learning, which is based on the idea of picking a random instance in the training set at every step and computing the gradient based only on that single instance.
2. The term "stochastic" means random, as opposed to "batch" gradient descent, which computes the gradient using the entire training set.
3. SGD is faster than batch gradient descent because it only needs to consider one instance at a time, making it more computationally efficient.
4. However, because SGD is based on random selection of instances, it is less regular than batch gradient descent. This means that the algorithm is more likely to jump out of local minima and have a better chance of finding the global minima.
5. The downside of SGD is that it may not settle at the minimum solution, as the algorithm is always searching for a better solution. To overcome this, a learning scheduling technique can be used, which gradually decreases the learning rate over time to help the algorithm converge to the optimal solution.


![sgd](https://github.com/SumitxThokar/Machine-Learning-Journey/blob/main/images/sgd.jpg)

## Day 35 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I delved more into Training models. I got the insights on Mini Batch Gradient Descent, Polynomial Regression and I also implemented the code using scikit-learn libraries. Mini-batch gradient descent (Mini Batch GD) is a powerful optimization technique that computes the gradient of a model's parameters on small, randomly selected subsets of training instances. And Polynomial Regression is a method where we fit a curve to the data instead of fitting a straight line.


![](https://github.com/SumitxThokar/Machine-Learning-Journey/blob/main/images/first.png)

## Day 36 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I learned about learning curves, a useful tool for evaluating whether a model is overfitting or underfitting. By plotting the model's performance on the training and validation sets, we can determine if the model is learning well or if it is struggling. I also learned that using a lower degree linear polynomial model reduces the chance of overfitting. In addition, I discovered Ridge Regression, a variation of linear regression that aims to keep model weights small by adding a regularized term to the cost function. The alpha parameter in Ridge Regression controls the degree of regularization applied to the model.

## Day 37 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I learned about Lasso Regression; another regularized version of Linear Regression. Here Lasso stands for Least Absolute Shrinkage and Selection Operator. It tends to completely eliminates the weights of the least important features meaning automatically performs feature selection and outputs a sparse model. We can use Lasso regression with scikit learn from linear_model with alpha parameter.

## Day 38 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I learned about Elastic Net which is middle ground between Ridge Regression and Lasso Regression. Further I learned when to use which Regression. It is always best to have atleast a little bit of regularization i.e to avoid Linear Regression. We should consider using Ridge regression by default but if you think only few features are useful, It is good to use lasso or ElasticNet. Furthermore, I learned about Early Stopping which basically is to stop training as soon as the validation error reaches as minimum to regularize iterative learning algorithms like Gradient Descent.

## Day 39 of #66daysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I learned what Logistic Regression is in detail and about its training and cost function. Logistic Regression is a model that estimates the probability of an event happening. The output ranges from 0 and 1. We use a cost udnciton that penalizes the model for making incorrect prediction which is called logloss and is calculated by averaging the cost over all the training data.

## Day 40 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning,Today I learned about Softmax Regression and how it can be used to generalize Logistic Regression models for multi-class classification. The process involves computing the exponentials of scores and then normalizing them. Also I gained more insignts on cross entropy whihc is used for minimizing cost function. It penalizes the model when it estimates a low probability for target class. 

### Implementation of Softmax Regression on scikit-learn
![](https://github.com/SumitxThokar/66DaysofML/blob/main/images/day39.jpg)

## Day 41 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today, I delved into the fascinating world of Support Vector Machines, a powerful machine learning model capable of performing linear and non-linear classification, regression, and outlier detection. With its ability to handle complex and small/medium-sized datasets, it's the go-to choice for many data scientists. However, it's worth noting that SVMs are sensitive to feature scale, and feature scaling can yield better results. I also learned about soft margin classification, which can handle noisy data but can be sensitive to outliers. In addition, I discovered that SVM models can be regularized by reducing C if they overfit the training data. Finally, I explored non-linear classification and the use of polynomial kernels in SVMs, expanding my knowledge of this powerful tool.

## Day 42 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I got insights on Gaussian Redial Basis Function (RBF) Kernel and kernels in general. Gaussian RBF i shighly flexible kernel that can approximate any continious function to arbitary accuracy. It is a popular choice for nonlinear regerssion and classification tasks. It can handle large data better than Polynomial Kernels. It is very useful when underly function is unknown or highly complex especially for cases like pattern recognition. I learned to implement it using sklearn. And The main thing to consider while training model is if the model is overfitting, reduce the gamma parameter and viceversa. Furthermore, I compared teh scikitlearn classes for SVM classification where I found out that SVC class takes the longest time to train while kernal tricks are only available in the same class.

## Day 43 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today, I delved deeper into the fascinating world of SVM Regression. As we know, Support Vector Machines (SVMs) are incredibly versatile as they support both classification and regression, whether the data is linear or nonlinear. I had the pleasure of implementing both Linear SVM Regression and Nonlinear Regression using Scikitlearn, which was an enriching experience.


## Day 44 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I got to learn about Decision Trees. Like SVM, Decision tree is capable of performing both classfication and regression tasks and even multioutput tasks. It is a very powerful algorithmn that can fit complex datasets. It can also be called the fundamental componenet of Random forest which is the most powerful ML algo right now. I trained Decision Tree model with Scikit-Learn using Iris dataset. I will be learning CART training algo and more about how to regularize them for next days.

## Day 45 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I got introduced to The CART(Classification and Regresion Tree) Training Alogrithm, Regularization Hyperparameters in Decision Trees and lastly learned how to implement regression problems using Decision Trees.
The CART training algo is the way how Decision tree works. First it splits the training set into two using single feature(k) and threshold(tk). The purest subsets of (k,tk) is taken and the same logic is implemented recursively until it reacher the max_depth which we can set using max_depth parameter. Talking about Decision Tree if we give it freedom, the tree most of the time overfits to the training data. To minimize overfitting we can use max_depth, min_samples_leaf and other parameters. 

## Day 46 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today was an interesting day as I got to learn about Ensemble Learning and Random Forests. Ensemble Learning is all predictions of group of predictors. Obviously the group of predictors will have better predictions than individual which will help to classify classes with better accuracy. The one class with highest votes is considered as the required class of an instance. Training group of Decision Tree classifier in different random subset of training set is Random Forest which is still one of the most powerful machine learning algorithm. Further more, I got introduced to voting classifiers which affregate the predicitons of each classifier and predict the class that gets most votes and they are of two types; Hard and soct voting.

## Day 47 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I was introduced to Bagging and Pasting. Here same classifier is used with random subsets. In sampling we perform subsets instance with replacement and exact opposite with pasting. Only bagging allows baggin allows training instances to be sampled several time for same predictor. And lastly I implemented the learning with scikit Learn.

## Day 48 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I learned about Out of Bag evaluation. Here Out of Bag means the training instances that are not trained. On average 33% of the the training instances are not trained for any classifiers. Since the predictors has not seen those instances before we can use them to evaluate evaluate without the need for validation. I implemented this learning using scikit Learn by setting **oob_score** True. We can also see the decision function for each training instance suing **oob_decision_function_**. Furthermore, I got insights on Random Pathces and Random Subspaces which is the concept of sampling the features. We can use this with **max_features** and **bootstrap_features** parameters. It is very useful when dealing with high dimension inputs like images. 

## Day 49 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I gained insights on Random Forests. Random Forest is the ensemble of Decision Trees. I implemented the Random Forest using Scikit-Learn. We can increase the random more by using random thresholds for each feature. It is called Extremely Randomized Trees (Extra-Trees). And also implemented scikit learn's class ExtraTreesClassifier. Furthermore, I learned about Feature Importance. It is easy to measure the relative importance of each feature. It is very handy to get a quick understanding of what features actually matter.


## Day 50 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I got insights on boosting which combines several weak learners into a strong learner. I t is an iterative process where we train predictores sequentially, each trying to correct its predecessors. The most popular Boosting method is Adaboost ie Adaptive Boosting. It is handly for binary classification. It pays attention to training instances and work out on what predecssor underfits. It gives higher weights to the example which are classified wrong so that the next classifier can focus more on the difficult example( the one that are classified wrongly). I implemented it with scikit Learn where I used SAAME method which is multiclass version of AdaBoost.

## Day 51 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I learned more on boosting methods. I got to learn about Gradient Boosting, XGBoost and Stacking. Gradient Boosting, similar to adaboost instead it tries to fit the new predictors to the residual errors made by previous predictor. XGBoost is a scalable, distributed gradient-boosted decision tree (GBDT) machine learning library. In stacking instead of using trival functions (hard voting) to aggragate the predictions of all predictors we train a model to perform this aggrigation. Unfortunately Scikit Learn doesnot support stacking. We can use brew which is an opensource library to implement it.

## Day 52 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I gained insights on Dimensionality Reduction, Dimensionality Reduction approaches like Projection and Manifold Learning. Projection is the all about reducing number of dimension in dataset by projecting into lower dimensional spaces. It simplify complex data and can improve efficiency. But it may lead to loss of information in dataset leading reduce in accuracy. Wrong choice in projection method leads to suboptimal result. Manifold learning is about reducing the number of features or variables in a dataset while retaining as much of the relevant information as possible where we have high dim dataset.

## Day 53 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I delved deeper into dimension reduction approaches. I got to learn about Principal Component Analysis (PCA). It is the most popular approach of Dimension Reduction. First we identify the hyperplane closest to the data and project them accordingly. We select the axis which preserve the maximum amount of variance as it loses less information of the data. Then I implemented these learning using Scikit Learn. I learned further more about explained Variance Ratio, How to choose the right number of dimension and to recover reduced data using inverse_tranform method.

## Day 54 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, I got insights on Randomized PCA, Incremental PCA, Kernel PCA and used GridSearch to select kernel and fine tune the best parameters. We implement Randomized PCA in scikit learn by setting svd_solver to randomized. Incremnetal PCA split the training set to mini barches and feed on IPCA algorithm. Kernel PCA is used to perform non linear complex projection for dimensionality reduction.

## Day 55 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I got insights on LLE: Locally linear embedding which is very powerful nonlinear dimensionality reduction technique. I implemented this technique using scikit learn. Furthermore, I explored more on Unsupervised learning Technique and I learned aboyt Clustering in detail. Clustering is the process of grouping similar data points together in order to identify patterns and gain insights from data. 

## Day 55 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, I got introduced to Kmeans. It is a simple clustering algorithm, which is quick and efficient. I implemented it using scikit learn. I also got to learn about transform() function which is used to measure the distance from each instance to every centroid. Furthermore, I got introduced to The k-means algorithm. It is a clustering technique that partitions data into k clusters based on their similarity. The algorithm begins by randomly initializing k centroids and assigning each data point to the closest centroid based on distance. The algorithm then recalculates the centroids for each cluster based on the mean position of the points in the cluster. The process of assigning points to clusters and recalculating centroids is repeated until convergence, when the assignments of data points to clusters no longer change significantly. The result is k clusters with their respective centroids and each data point assigned to the cluster with the closest centroid.

## Day 56 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I delved more deeper into the world of KMeans. I started the day learning about Accelerated Kmeans and MiniBatch KMeans and learned how to import the Minibatch Kmeans. Next I got to learn about the techniques to find the optimal number of cluster in the dataset. I got introduced to elbow technique where we map inertia against k (number of cluster) and Silhoutte score where we compare Silhoutte score with various values of k. Next thing I learned was the limits of KMeans. KMeans donot behave well when cluster have varying sizes, different densities and non-spherical shape. Furthermore I learned how we can use cluster for Image segmentation, for preprocessing, for semi-supervised learning etc. 

## Day 57 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I got insights on DBSCAN. DBSCAN is a clustering algorithm that can identify clusters and noise points without needing to know the number of clusters in advance. Its advantages include its ability to handle arbitrary shapes and sizes, and its ability to automatically determine the number of clusters. However, it may not perform well on datasets with varying densities, and its performance can be sensitive to parameter choices. Additionally, it may not be suitable for high-dimensional datasets. 

## Day 58 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I got to learn about Gaussian mixture model. Gaussian mixture models (GMMs) are statistical models that represent a dataset as a mixture of Gaussian distributions. GMM estimates the parameters of the Gaussian distributions and the weights of each distribution in the mixture using the Expectation-Maximization (EM) algorithm. The EM algorithm iteratively computes the probabilities of each data point belonging to each Gaussian distribution and updates the mean, variance, and weight parameters of each distribution based on those probabilities until convergence. GMM can be used for density estimation, classification, and data generation.

## Day 59 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I got my foot on Artificial Neural Networks. It is a computational model. It is based on the structure and functions of biological neural networks. Firstly I got introduced to the historical background of ANNS like how it was inspired from biological neurons and all. I got insights on ANNs being at the core of Deep Learning and learned about logical computations stored with neurons. Later on I got introduced to Pereptrons which is one of the simplest ANN architecture which is also very similar to SGD. I also got to learn that stacking these perceptrons (Multilayered) significantly overcomes many shortcoming in perceptrons with single layer.

## Day 60 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I learned about Backpropagation algorithm, popular activation function and why activation function is used. Backpropagation can be split into 3 parts: forward pass, for each instance the algo makes a prediction. reverse pass, measure the error and goes through each layer in reverse to measure the error contribution from each connection. And Gradient Descent step, slightly tweks the connection cuts to reduce the error. Next, I learned about hyperbolic tangent function tanh and Rectified Linear Unit function ReLU. Finally learned activation function are used to solve very complex problem. 

## Day 61 of #66DaysofML.
**Course**:**Hands on Machine Learning with scikit Learn, Keras & TensorFlow** <br>
:large_blue_diamond: On my journey of learning Machine Learning, Today I learned about classification and learned to train and implement MLPs with Keras. A Multi-Layer Perceptron (MLP) is a type of feedforward neural network commonly used for classification tasks. It consists of an input layer, one or more hidden layers, and an output layer. To implement an MLP with Keras, you can follow these steps:
- Import the necessary libraries.
- Define the model using the Sequential API.
- Compile the model.
- Fit the model to the training data.
