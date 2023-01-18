# **Machine Learning**
![MachineLearning](https://github.com/SumitxThokar/MachineLearning/blob/main/images/pexels-tara-winstead-8386434.jpg)

## Day 1 of #66DaysofML
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
:large_blue_diamond: On my journey of learning Learning ML. These are my finding and learnings today:
### Reducing Loss<br>
Achieving optimal model performance by iteratively adjusting parameters to reduce loss. A Machine Learning model is trained by starting with an initial guess for the weights and bias and iteratively adjusting those guesses until learning the weights and bias with the lowest possible loss.
![Reducing Loss](https://github.com/SumitxThokar/MachineLearning/blob/main/images/Iterative.png)
### Gradient Descent
Gradient descent is an optimization algorithm used to minimize a function by iteratively adjusting the parameters in the direction of the negative gradient of the function. It is a widely used algorithm in machine learning to find the best parameters of the model that minimize the loss. The gradient always points in the direction of steepest increase in the loss function. The gradient descent algorithm takes a step in the direction of the negative gradient in order to reduce loss as quickly as possible.
![Gradient Descent](https://github.com/SumitxThokar/MachineLearning/blob/main/images/gd.png)

## Day 4 of #66DaysofML.
### Learning Rate
:large_blue_diamond: The Gradient Descent algorithm is a powerful tool for optimizing machine learning models by adjusting the parameters of the model in order to minimize the error. One of the key components of this algorithm is the **learning rate**, which is a scalar value that is multiplied by the gradient in order to determine the next point in the optimization process. For example, if the gradient magnitude is 2.5 and the learning rate is 0.01, the algorithm will move to a point that is 0.025 units away from the previous point. This small step size allows for a gradual and controlled optimization process. In addition, gradient descent can also be applied to feature sets that contain multiple features, making it a versatile algorithm for a wide range of machine learning tasks. 
<br> **When the Learning rate is small.**
![Small Learning rate](https://github.com/SumitxThokar/MachineLearning/blob/main/images/rate.png)
<br>**What Learning rate should be like.**
![Good Learning rate](https://github.com/SumitxThokar/MachineLearning/blob/main/images/goodrate.png)
<br>A variant of this algorithm is **Stochastic Gradient Descent** which uses random samples from the data to update the parameters.

## Day 5 of #66DaysofML.
:large_blue_diamond: Today, I was introduced to TensorFlow, an open-source platform for machine learning. TensorFlow's APIs are organized in a hierarchical structure, with high-level APIs built on top of low-level APIs. As a beginner, I will be using the high-level APIs, specifically Keras, which simplifies the process of building and training machine learning models. I revisited my knowledge of NumPy and pandas to better understand the code written in tf.keras. I also became familiar with the use of Google Colab, a tool that allows for easy collaboration and code execution. One of the main objectives of the day was to learn how to use linear regression code in tf.keras. I also evaluated loss curves and practiced tuning hyperparameters to improve the performance of my model. Overall, it was a valuable experience and I look forward to continuing my learning journey with TensorFlow.
![Summary](https://github.com/SumitxThokar/MachineLearning/blob/main/images/summarytuninghyperparameter.jpg)

## Day 6 of #66DaysofML.
:large_blue_diamond: Today, I got to develop intuition about overfitting and got idea on how to determine whether a model is good or not.<br>
In machine learning, it is important to consider the ability of a model to generalize to new, unseen data. This is known as generalization. On the other hand, overfitting occurs when a model is too complex and has learned noise from the training data, rather than the underlying pattern. An overfit model will have a low loss during training, but will perform poorly on new data.<br>
So, how can we determine whether a model is good or not? One way is to evaluate its performance on new data. To do this, we can divide our dataset into a training set and a test set. The training set is used to fit the model, while the test set is used to evaluate the model's performance. If the model performs well on the test set, it is an indicator that the model is good.<br>
In summary, generalization and overfitting are important concepts in machine learning. To determine whether a model is good, we can evaluate its performance on new data by using a train and test set. A good performance on the test set is an indicator of a good model.
<br>
## Day 7 of #66DaysofML.
:large_blue_diamond: When it comes to building a machine learning model, one of the most important steps is dividing your data set into a training set and a test set. This is because having separate data sets for training and testing allows you to gauge the performance of your model in a more accurate and reliable way.
<br>
![Training and Test data](https://github.com/SumitxThokar/MachineLearning/blob/main/images/Trainingtest.jpg)
The training set is used to train your model, and the test set is used to evaluate the performance of the model once it has been trained. The larger the training set, the better your model will learn and the more accurate it will be. This is because a larger training set provides the model with more data to learn from and generalize from.<br>
On the other hand, the larger the test set, the more confident you can be in the performance of your model. This is because a larger test set provides a more representative sample of the data and can give you a better idea of how well your model will perform on unseen data.
<br>
However, it is important to keep in mind that when working with smaller data sets, you may need to use cross-validation techniques to effectively divide your data into training and test sets. Cross-validation is a statistical method that allows you to train and test your model on different subsets of your data, giving you a more robust estimate of the model's performance.
<br>
It is also important to note that it is never a good idea to train your model on the test data. This is because the test data is meant to be used as a way to evaluate the model's performance on unseen data, and training on this data would lead to an overestimation of the model's performance.
<br>
## Day 8 of #66DaysofML.
Dividing your data into training, validation, and test sets is a common practice in machine learning to ensure that the model is properly validated and generalizes well to new data. The first step in this workflow is to train the model on the training set, which allows it to learn the patterns and relationships in the data.<br>
![Validation set](https://github.com/SumitxThokar/MachineLearning/blob/main/images/validation_set.jpg)
The next step is to evaluate the model on the validation set, which is a set of data that the model has not seen before. This allows you to see how well the model is performing on new data and to adjust the hyperparameters, such as the learning rate or the number of hidden layers, accordingly. Once you have found the best performing model on the validation set, you can then confirm its performance on the test set, which is a final set of unseen data. This workflow greatly reduces the chance of overfitting, which occurs when a model is overly optimized for the training data and performs poorly on new data.<br>
![Workflow](https://github.com/SumitxThokar/MachineLearning/blob/main/images/better_workflow.jpg)
