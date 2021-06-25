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
[image18]: assets/19.png 
[image20]: assets/20.png 


# Introduction to Neural Networks

How does Neural Networks work?

## Content 
- [Feed forward - Weighted sum - Boundaries](#bound)

- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)

## Feed forward - Weighted sum - Boundaries <a name="bound"></a>
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

### Nonlinear Regions
- Combine Perceptrons to create nonlinear regions

    ![image5]

### More Nodes/Layers
- 2 possibilities: 
    - more nodes
    - more hidden layers

    ![image6]

### Feed Forward
- Check out below some examples of Perceptron Feed Forward

    ![image7]

### Activation Functions
- Source [Wikipedia](https://en.wikipedia.org/wiki/Activation_function) 

    ![image8]
    ![image9]

### Sigmoid Activation for binary probabilities 
- Switch from discrete to continuous via Sigmoid Activation

    ![image11]

### Maximum Likelihood
- Use a model that gives the existing labels the highest probability
- If P(all) is high than: The model classifies most points correctly with P(all) indicating how accurate the model is

### Two important Error Functions (cost functions)
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





### Perceptron Trick
- Perceptron model Line should move closer to a misclassified point

    ![image3]


### Perceptron algorithm
- Update weights and bias for every **misclassified** point

    ![image4]

### Gradient Descent algorithm
- Update weights and bias for **every** point

    ![image13]


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
