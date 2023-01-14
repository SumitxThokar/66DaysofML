# **Machine Learning**
![MachineLearning](https://github.com/SumitxThokar/MachineLearning/blob/main/images/pexels-tara-winstead-8386434.jpg)

## Day 1 of #66DaysofML
I am starting my machine learning with **Google Machine Learning Crash Course**.<br>
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
Onto the journey of learning ML, Today I relate weights and biases in machine learning to slope and offset in line fitting, understand "loss" in general and squared loss in particular.<br>
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
### Reducing Loss<br>
Achieving optimal model performance by iteratively adjusting parameters to reduce loss. A Machine Learning model is trained by starting with an initial guess for the weights and bias and iteratively adjusting those guesses until learning the weights and bias with the lowest possible loss.
![Reducing Loss](https://github.com/SumitxThokar/MachineLearning/blob/main/images/Iterative.png)
### Gradient Descent
Gradient descent is an optimization algorithm used to minimize a function by iteratively adjusting the parameters in the direction of the negative gradient of the function. It is a widely used algorithm in machine learning to find the best parameters of the model that minimize the loss. The gradient always points in the direction of steepest increase in the loss function. The gradient descent algorithm takes a step in the direction of the negative gradient in order to reduce loss as quickly as possible.
![Gradient Descene](https://github.com/SumitxThokar/MachineLearning/blob/main/images/gd.png)

## Day 4 of #66DaysofML.
### Learning Rate
The Gradient Descent algorithm is a powerful tool for optimizing machine learning models by adjusting the parameters of the model in order to minimize the error. One of the key components of this algorithm is the **learning rate**, which is a scalar value that is multiplied by the gradient in order to determine the next point in the optimization process. For example, if the gradient magnitude is 2.5 and the learning rate is 0.01, the algorithm will move to a point that is 0.025 units away from the previous point. This small step size allows for a gradual and controlled optimization process. In addition, gradient descent can also be applied to feature sets that contain multiple features, making it a versatile algorithm for a wide range of machine learning tasks. 
<br> **When the Learning rate is small.**
![Small Learning rate](https://github.com/SumitxThokar/MachineLearning/blob/main/images/rate.png)
<br>**What Learning rate should be like.**
1[Good Learning rate](https://github.com/SumitxThokar/MachineLearning/blob/main/images/goodrate.png)
<br>A variant of this algorithm is **Stochastic Gradient Descent** which uses random samples from the data to update the parameters.
