[image1]: assets/1.png 
[image2]: assets/2.png 
[image3]: assets/3.png 
[image4]: assets/4.png 
[image5]: assets/5.png 
[image6]: assets/6.png 
[image7]: assets/7.png 
[image8]: assets/8.png 
[image9]: assets/9.png 
[image10]: assets/10.png 
[image11]: assets/11.png 
[image12]: assets/12.png 
[image13]: assets/13.png 
[image14]: assets/14.png 
[image15]: assets/15.png 
[image16]: assets/16.png 
[image17]: assets/17.png 
[image18]: assets/18.png 
[image19]: assets/19.png 
[image20]: assets/20.png 
[image21]: assets/21.png 
[image22]: assets/22.png 
[image23]: assets/23.png 
[image24]: assets/24.png 
[image25]: assets/25.png 
[image26]: assets/26.png 
[image27]: assets/27.png 
[image28]: assets/28.png 
[image29]: assets/29.png 
[image30]: assets/30.png 


# Neural Networks

How does Neural Networks work?

# Content 
- [Machine Learning algorithm](#ml_algo)
- [Model - Feed forward - Weighted sum - Boundaries](#bound)
    - [Boundary concepts](#bound_concepts)
    - [Perceptron](#perceptron)
    - [Perceptron Trick](#percep_trick)
    - [Perceptron algorithm](#percep_algo)
    - [Nonlinear Regions](#nonlinear_regions)
    - [More Nodes/Layers](#more_nodes_layers)
    - [Activation Functions](#activation_func)
    - [Sigmoid Activation for binary probabilities](#sigmoid_activation)
    - [Softmax Activation for multiclass probabilities](#softmax_activation)
    - [One Hot Encoding](#one_hot)
    - [Maximum Likelihood](#max_likely)
    - [Important Error Functions](#error_func)

- [Gradient Descent](#gradient_descent)
    - [Gradient Descent as optimization algorithm](#grad_dec)
    - [Gradient Descent algorithm](#grad_dec_algo)
    - [Forward and Backward propagation](#for_and_back)
    - [Minimal Example in NumPy](#min_example)
- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)

# Machine Learning algorithm <a name="ml_algo"></a>
- A machine learning algorithm can be thought of as a **black box**. 
- It takes **inputs** and gives **outputs**.
- Once we have a **model**, we must **train** it. Training is the process through which, the model learns how to make sense of input data.

    ![image15]

    [Source: Udemy - The Data Science Course 2021: Complete Data Science Bootcamp ](https://www.udemy.com/course/the-data-science-course-complete-data-science-bootcamp/)

- Types of machine/deep learning

    ![image16]

    [Source: Udemy - The Data Science Course 2021: Complete Data Science Bootcamp ](https://www.udemy.com/course/the-data-science-course-complete-data-science-bootcamp/)

- The Building blocks of a machine learning algorithm

    ![image17]

    [Source: Udemy - The Data Science Course 2021: Complete Data Science Bootcamp ](https://www.udemy.com/course/the-data-science-course-complete-data-science-bootcamp/)

- Regression vs Classification

    Supervised learning could be split into two subtypes – regression and classification. Regression

    ![image18]

    [Source: Udemy - The Data Science Course 2021: Complete Data Science Bootcamp ](https://www.udemy.com/course/the-data-science-course-complete-data-science-bootcamp/)

# Model - Feed forward - Weighted sum - Boundaries <a name="bound"></a>
## Model <a name="model"></a> 
- The simplest possible model is a **linear model**. It is the basis of more complicated (nonlinear) models.

    ![image19]

- Example:

    ![image20]

## Boundary concepts <a name="bound_concepts"></a> 
- **2D: Linear Boundary**
    - 2 inputs
    - 2 dimensions
    - Linear line is the model
    - It is the Boundary between the two classes

- **3D Plane Boundary**
    - 3 inputs
    - 3 dimensions
    - Plane as the model
    - It is the Boundary between the three classes

- **Higher Dimensions**
    - n inputs
    - n dimensions
    - n-1 dimensional **hyperplane as the model**
    - It is the Boundary between the n classes

    ![image1]

## Perceptron <a name="perceptron"></a>
### Why 'Neural Networks'?
- Input Nodes = Dendrites
- Linear Function = Nucleus
- Activation Function = Axon 

    ![image2]

## Perceptron Trick <a name="percep_trick"></a>
- Perceptron model Line should move closer to a misclassified point

    ![image3]

## Perceptron algorithm <a name="percep_algo"></a>
- Update weights and bias for every **misclassified** point

    ![image4]

## Nonlinear Regions <a name="nonlinear_regions"></a>
- Combine Perceptrons to create nonlinear regions

    ![image5]

## More Nodes/Layers <a name="more_nodes_layers"></a>
- 2 possibilities: 
    - more nodes
    - more hidden layers

    ![image6]

## Activation Functions <a name="activation_func"></a>
- [Source: Udemy - The Data Science Course 2021: Complete Data Science Bootcamp ](https://www.udemy.com/course/the-data-science-course-complete-data-science-bootcamp/)

    ![image9]

## Sigmoid Activation for binary probabilities <a name="sigmoid_activation"></a>
- Switch from discrete to continuous via Sigmoid Activation

    ![image11]

## Softmax Activation for multiclass probabilities <a name="softmax_activation"></a> 
- The softmax activation transforms a bunch of arbitrarily large or small numbers into a valid probability distribution.
- Often used for final output layer
- However, when the softmax is used prior to that (as the activation of a hidden layer), the results are not as satisfactory. That’s because a lot of the information about the variability of the data is lost.

    ![image23]

    [Source: Udemy - The Data Science Course 2021: Complete Data Science Bootcamp ](https://www.udemy.com/course/the-data-science-course-complete-data-science-bootcamp/)

## Maximum Likelihood <a name="max_likely"></a>
- Use a model that gives the existing labels the highest probability
- If P(all) is high than: The model classifies most points correctly with P(all) indicating how accurate the model is

## Important Error Functions (cost functions) <a name="error_func"></a>
- Cost functions must be:
    - continuous 
    - differentiable
- Mean Squared Eror (MSE)
    - Outliers will be penelized drastically. 
    - Difference between observation and prediction is always positive.
- Cross Entropy (CE)
    - If one factors is turning to 0 ---> P(all)=P(1)*P(2)*... is affected drastically.
    - Use sums instead of products.
    - Use log functions.
    - A bad model has a high cross entropy.
    - A good model has a low cross entropy.
    - A misclassified point has a large log value (error).
    - A correct classified point has a low log value (error).

    ![image10]

    ![image14]

    ![image12]

# Gradient Descent <a name="gradient_descent"></a>

## Gradient Descent as optimization algorithm <a name="grad_dec"></a>
- The last ingredient is the **optimization algorithm**. The most commonly used one is the **gradient descent**. The main point is that we can find the **minimum of a loss function** by applying the rule: 𝑥𝑖+1 = 𝑥𝑖 − 𝜂𝑓′(𝑥𝑖) , where 𝜂 is a small enough positive number. In machine learning, 𝜂, is called the learning rate. The rationale is that the first derivative at xi , f’(xi) shows the slope of the function at xi.

    ![image21]

    [Source: Udemy - The Data Science Course 2021: Complete Data Science Bootcamp ](https://www.udemy.com/course/the-data-science-course-complete-data-science-bootcamp/)

- Learning rate: Best Practice

    ![image22]

    [Source: Udemy - The Data Science Course 2021: Complete Data Science Bootcamp ](https://www.udemy.com/course/the-data-science-course-complete-data-science-bootcamp/)


## Gradient Descent algorithm <a name="grad_dec_algo"></a>
- Update weights and bias for **every** point

    ![image13]

## Forward and Backward propagation <a name="for_and_back"></a>
- Forward and backward propagation needed for learning 

    ![image24]

## Minimal example in NumPy <a name="min_example"></a>
- Open Jupyter Notebook ```NN_numpy.ipynb```
    ### Libraries
    ```
    import numpy as np
    ```
    ### Generate random input data to train on
    ```
    observations = 1000000
    xs = np.random.uniform(low=-10, high=10, size=(observations,1))
    zs = np.random.uniform(-10, 10, (observations,1))

    # Combine the two dimensions of the input into one input matrix. 
    # This is the X matrix from the linear model y = x*w + b.
    # column_stack is a Numpy method, which combines two vectors into a matrix. Alternatives are stack, dstack, hstack, etc.
    
    inputs = np.column_stack((xs,zs))
    print (inputs.shape)
    ```
    ### Generate the targets we will aim at
    ``` 
    noise = np.random.uniform(-1, 1, (observations,1))

    # Produce the targets according to the f(x,z) = 2x - 3z + 5 + noise definition.

    targets = 13*xs + 7*zs - 12 + noise
    print (targets.shape)
    ```
    ### Initialize weights 
    ```
    # init_range is the variable to set a range for initialization.
    init_range = 0.1

    weights = np.random.uniform(low=-init_range, high=init_range, size=(2, 1))

    biases = np.random.uniform(low=-init_range, high=init_range, size=1)

    print (weights)
    print (biases)
    ```
    ### Set a learning rate
    ```
    learning_rate = 0.02
    ```
    ### Train the model
    ```
    for i in range (100):
        
        deltas = outputs - targets
        loss = np.sum(deltas ** 2) / 2 / observations
       
        # Another small trick is to scale the deltas the same way as the loss function. In this way our learning rate is independent of the number of samples (observations).
        
        deltas_scaled = deltas / observations
        
        # The gradient descent update rule
        # weights            2x1 
        # bias               1x1 (scalar)
        # learning_rate      1x1 (scalar)
        # inputs             1000x2
        # deltas_scaled      1000x1
        # outputs            1000x1
        # targets            1000x1
        # We must transpose the inputs so that we get an allowed operation.

        weights = weights - learning_rate * np.dot(inputs.T,deltas_scaled)
        biases = biases - learning_rate * np.sum(deltas_scaled)
    ```

## Setup Instructions <a name="Setup_Instructions"></a>
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites: Installation of Python via Anaconda and Command Line Interaface <a name="Prerequisites"></a>
- Install [Anaconda](https://www.anaconda.com/distribution/). Install Python 3.7 - 64 Bit

- Upgrade Anaconda via
```
$ conda upgrade conda
$ conda upgrade --all
```

- Optional: In case of trouble add Anaconda to your system path. Write in your CLI
```
$ export PATH="/path/to/anaconda/bin:$PATH"
```

### Clone the project <a name="Clone_the_project"></a>
- Open your Command Line Interface
- Change Directory to your project older, e.g. `cd my_github_projects`
- Clone the Github Project inside this folder with Git Bash (Terminal) via:
```
$ git clone https://github.com/ddhartma/Matrix-Math-with-Numpy.git
```

- Change Directory
```
$ cd Matrix-Math-with-Numpy
```

- Create a new Python environment, e.g. matrix_op. Inside Git Bash (Terminal) write:
```
$ conda create --name matrix_op
```

- Activate the installed environment via
```
$ conda activate matrix_op
```

- Install the following packages (via pip or conda)
```
numpy = 1.12.1
pandas = 0.23.3
matplotlib = 2.1.0
seaborn = 0.8.1
```

- Check the environment installation via
```
$ conda env list
```

## Acknowledgments <a name="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Deep Reinforcement Learning'. Please check this [link](https://www.udacity.com) for more information.

## Further Links <a name="Further_Links"></a>

Git/Github
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
* [5 types of Git workflows](https://buddy.works/blog/5-types-of-git-workflows)

Docstrings, DRY, PEP8
* [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/)
* [DRY](https://www.youtube.com/watch?v=IGH4-ZhfVDk)
* [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

## Further Links <a name="Further_Links"></a>
Git/Github
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
* [5 types of Git workflows](https://buddy.works/blog/5-types-of-git-workflows)

Docstrings, DRY, PEP8
* [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/)
* [DRY](https://www.youtube.com/watch?v=IGH4-ZhfVDk)
* [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

Important web sites - Deep Learning
* Deep Learning - illustriert - [GitHub Repo](https://github.com/the-deep-learners/deep-learning-illustrated)
