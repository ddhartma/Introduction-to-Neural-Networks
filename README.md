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
[image31]: assets/31.png 
[image32]: assets/32.png 
[image33]: assets/33.png 
[image34]: assets/34.png 
[image35]: assets/35.png 
[image36]: assets/36.png 
[image37]: assets/37.png 
[image38]: assets/38.png 
[image39]: assets/39.png 
[image40]: assets/40.png 
[image41]: assets/41.png 
[image42]: assets/42.png 
[image43]: assets/43.png 
[image44]: assets/44.png 
[image45]: assets/45.png 
[image46]: assets/46.png 
[image47]: assets/47.png 
[image48]: assets/48.png 
[image49]: assets/49.png 
[image50]: assets/50.png 
[image51]: assets/51.png 
[image52]: assets/52.png 
[image53]: assets/53.png 
[image54]: assets/54.png 
[image55]: assets/55.png 
[image56]: assets/56.png 
[image57]: assets/57.png 
[image58]: assets/58.png 
[image59]: assets/59.png 
[image60]: assets/60.png 
[image61]: assets/61.png 

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
    - [Digging Deeper](#dig_deeper)
    - [Why do we need nonlinearities?](#why_nonlinear) 
    - [Activation Functions](#activation_func)
    - [Sigmoid Activation for binary probabilities](#sigmoid_activation)
    - [Softmax Activation for multiclass probabilities](#softmax_activation)
    - [Maximum Likelihood](#max_likely)
    - [Important Error Functions](#error_func)
- [Underfitting and Overfitting](#under_over)
    - [Ways to overcome Under - and Overfitting](#over_under_overcome)
        - [Train-Validation-Test Split](#train_val_test)
        - [N-fold-cross-validation](#n_fold_cross_val)
        - [Early Stopping](#early_stop)
        - [Dropout](#dropout)
        - [L1 and L2 regularization](#l1_l2)
        - [Data Augmentation](#data_aug)
- [Weight Initialization](#weight_init)
    - [Why is weight initialization important?](#why_weight_init)
    - [Why not a constant setting?](#not_constant)
    - [Problem of initializations with Random Uniform or Random Normal](#uniform_normal_init)
    - [Glorot normal/uniform distribution](#glorot_normal)
- [Instable Gradients](#instable_grad)
    - [Vanishing gradients](#van_grad)
    - [Exploding gradients](#expl_grad)
    - [Batchnormalization](#batchnorm)
- [Optimizer](#optimizer)
    - [(Batch) Gradient Descent](#gradient_descent)
        - [Gradient Descent as optimization algorithm](#grad_dec)
        - [Gradient Descent algorithm](#grad_dec_algo)
        - [Forward and Backward propagation](#for_and_back)
        - [Minimal Example in NumPy](#min_example)
    - [Stochastic Gradient Descent](#sgd)
    - [Minibatch Gradient Descent](#mini_batch)
    - [Momentum](#momentum)
    - [Learning Rate](#lr)
    - [Hyperparameters vs. Parameters](#hyper_para)
    - [AdaGrad and RMSProp](#adagrad)
    - [Adam](#adam)
- [Preprocessing](#preprocess)
    - [Why Preprocessing?](#why_preprocess)
    - [Types](#types)
        - [Relative Values](#rel_val)
        - [Logarithms](#loga)
        - [Normalization](#normal)
        - [Standardization](#standard)
        - [Normalization with L1 or L2 norm](#norma_l1_l2)
        - [Principal Component Analysis](#pca)
        - [Whitening](#white)
        - [Treating Categorical Data](#categorical)
        - [Data Augmentation](#data_aug)

- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)

# Machine Learning algorithm <a name="ml_algo"></a>
- A machine learning algorithm can be thought of as a **black box**. 
- It takes **inputs** and gives **outputs**.
- Once we have a **model**, we must **train** it. Training is the process through which, the model learns how to make sense of input data.

    ![image15]

    [Source: Udemy - The Data Science Course 2021: Complete Data Science Bootcamp ](https://www.udemy.com/course/the-data-science-course-complete-data-science-bootcamp/)

- Types of machine/deep learning:

    ![image16]

    [Source: Udemy - The Data Science Course 2021: Complete Data Science Bootcamp ](https://www.udemy.com/course/the-data-science-course-complete-data-science-bootcamp/)

- The Building blocks of a machine learning algorithm:

    ![image17]

    [Source: Udemy - The Data Science Course 2021: Complete Data Science Bootcamp ](https://www.udemy.com/course/the-data-science-course-complete-data-science-bootcamp/)

- Regression vs Classification:

    Supervised learning could be split into two subtypes – **regression** and **classification**. 

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
- Perceptron model Line should move closer to a misclassified point:

    ![image3]

## Perceptron algorithm <a name="percep_algo"></a>
- Update weights and bias for every **misclassified** point:

    ![image4]

## Nonlinear Regions <a name="nonlinear_regions"></a>
- Combine Perceptrons to create nonlinear regions:

    ![image5]

## More Nodes/Layers <a name="more_nodes_layers"></a>
- 2 possibilities: 
    - more nodes
    - more hidden layers

    ![image6]

## Digging Deeper <a name="dig_deeper"></a>
- This is a deep neural network (deep net) with 5 layers.

    ![image25]

## Why do we need nonlinearities? <a name="why_nonlinear"></a>
- Because **two consecutive linear transformations** are **equivalent** to a **single one**.

    ![image29]


## Activation Functions <a name="activation_func"></a>
- [Source: Udemy - The Data Science Course 2021: Complete Data Science Bootcamp ](https://www.udemy.com/course/the-data-science-course-complete-data-science-bootcamp/)

    ![image9]

## Sigmoid Activation for binary probabilities <a name="sigmoid_activation"></a>
- Switch from discrete to continuous outputs via sigmoid activation:

    ![image11]

## Softmax Activation for multiclass probabilities <a name="softmax_activation"></a> 
- The softmax activation transforms a bunch of arbitrarily large or small numbers into a valid probability distribution.
- It is often used for the final output layer.
- However, when the softmax is used prior to that (as the activation of a hidden layer), the results are not as satisfactory. That’s because a lot of the information about the variability of the data is lost.

    ![image23]

    [Source: Udemy - The Data Science Course 2021: Complete Data Science Bootcamp ](https://www.udemy.com/course/the-data-science-course-complete-data-science-bootcamp/)

## Maximum Likelihood <a name="max_likely"></a>
- Use a model that gives the existing labels the highest probability.
- If P(all) is high than: The model classifies most points correctly with P(all) indicating how accurate the model is.

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

# Underfitting and Overfitting <a name="under_over"></a>
- A good model captures the undelying logic of the dataset.
- However: There are two adverserial extremes:
    - Underfitting
    - Overfitting

    ![image30]

## Ways to overcome Under - and Overfitting <a name="over_under_overcome"></a> 
## Train-Validation-Test Split <a name="train_val_test"></a> 
- A Train-Validation split is needed to identify Underfitting or Overfitting.
- **Training set**: That's where the training happens.
- **Validation set**: Helps to prevent overfitting. 
- **Test set**: Measures the final predictive power of the model.

    ![image31]

    - Training is done only on the **Training set**.
    - Training: Apply Forward and Backward Propagation.
    - Apply 'Training' for a while.
    - Then switch to the **Validation set**.
    - Validation: Apply only Forward Propagation.
    - Calculate the loss: How does it perform?
        - Validation and Training loss should decrease during Training.
        - At some point the Validation loss could start increasing.
        This is the point where overfitting starts. 
        - This is the point where one should stop training the model.
    - Perform Training and Validation iteratively.
    - After training: One should test the model performance using the **Test set** (only Forward Propagation).

## N-fold-cross-validation <a name="n_fold_cross_val"></a> 
- It is useful if the Dataset is very small.

    ![image32]

    - It combines Training and Validation set in a clever way.
    - For example: we have 10000 observations (9000 for training + 1000 for validation).
    - Divide dataset into **n chunks** (here 10).
    - During the **1st epoch**: **1st chunk** serves as the **validation set**.
    - During the **2nd epoch**: **2nd chunk** serves as the **validation set**.
    - etc.
    - For each epoch: We don't overlap training and validation.

    
## Early Stopping <a name="early_stop"></a> 
- An easy method to prevent overfitting.

    ![image33]

    **Benefits**:
    - We are sure the validation loss is minimized.
    - Saves computing power.
    - Prevents overfitting.

## Dropout <a name="dropout"></a> 
- An easy approach to reduce the problem of overfitting.

    ![image34]

    - Each node will be dropped out with a certain propability, e.g. of 20%.
    - Consider: During validation or testing we do not want to use Dropout!!! We want to use the full predictive power of the full net.
    - However: Due to an increased number of available parameters during validation/testing (when dropout has been used during training), you have to reduce the neuron parameter. For example for a dropout probability of 20%, multiply the parameters of each layer with 0.8 (automatically done in Keras and PyTorch).

- Simple rules for Dropout:
    - Use it in case of overfitting.
    - Even without overfitting it show an performance increase during validation. Test it e.g. in **later training epochs**.
    - Rule of thumb: Test it first in the last hidden layer. Positive effect? If yes, test it in the second last, and so on ...
    - Too much Dropout could be counterproductive.
    - Good starting point: 20%...50%

## L1 and L2 regularization <a name="l1_l2"></a>  
- An easy approach to reduce the problem of overfitting.

    ![image35]

    - In this way only parameters (weights) will be kept if they lead to a significant reduction between target and prediction values (reduction of loss).

## Data Augmentation <a name="data_aug"></a>  
- An easy approach to reduce the problem of overfitting.
- Popular transformations (often used for images):
    - Flip
    - Rotation
    - Scale
    - Crop
    - Translation
    - Gaussian Noise 

# Weight Initialization <a name="weight_init"></a> 
## Why is weight initialization important? <a name="why_weight_init"></a> 
- To overcome local minima a random restart could beneficial. 
- The initial value of weights is important for learning.
- A deep neural network uses activation functions like sigmoid to implement nonlinearities.
- If **all weights are around zero**, the sigmoid will produce **linear outputs** (linear region of sigmoid is around 0). The Deep Neural Network will **loose depth**.
- If **all weights are too negative or to positive**, the sigmoid is almost **flat** (the sigmoid outputs are 1 or 0 then). A **Vanishing Gradient** problem will arise --> The **network will not learn**.

## Why not a constant setting? <a name="not_constant"></a> 
- If all weights are equal there is no reason for the algorithm to learn.

    ![image36]

    - In Forward Propagation: 
        - h<sub>1</sub>=h<sub>2</sub>=h<sub>3</sub>=h
        - y<sub>1</sub>=y<sub>2</sub>=y
    - Some optimization will take place but weights remain useless.


## Problem of initializations with Random Uniform or Random Normal   <a name="uniform_normal_init"></a> 
- Those initializations support more the linear region than nonlinear regions of the of sigmoid function.

    ![image38]

## Glorot normal/uniform distribution <a name="glorot_normal"></a>  
- Read the original paper of Xavier Glorot, Yoshua Bengio, [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf), Proceedings of the Thirteens International Conference on Artificial Intelligence and Statistics, PMLR 9:249-256, 2010.

    ![image39]
    - This initialization is exactly what we need, i.e. a wide range of initialized weights.
    - Two types of initializations:
        - Uniform Glorot initialization
        - Normal Glorot initialization
    - Main idea: The **number of neurons in the input and output layer** is important for the **range of the weight values**.
    - Why are the number of input and output neurons so important?
        - The more outputs the higher the need to spread weights.
        - Inputs are the 'Outputs' for Backpropagation.
    - In Tensorflow the Glorot Uniform intitializer (glorot_uniform_initializer) will be used as default.

# Instable Gradients <a name="instable_grad"></a> 
- Instable Gradients of neurons can be problematic if we add further hidden layers.
- There are two types of instable gradients:
    - Vanishing Gradients
    - Exploding Gradients

## Vanishing gradients <a name="van_grad"></a> 
- A weight will be changed based on the gradient size of the cost function.  
- If the gradient is 0 --> no learning (weight update). 
- Weight update is most effective for the hidden layer close to the output layer.
- For hidden layers further away from the output layer --> gradient of the cost function will become more flat. --> Vanishing learning effect.

## Exploding gradients <a name="expl_grad"></a> 
- Increase of gradients is steeper for hidden layers further away from the output layer.
- This is often the case for Recurrent Neural Networks.
- They decrease the learnability of neural networks as they saturate neurons.

## Batchnormalization <a name="batchnorm"></a> 
- Batch-Normalization (BN) is an algorithmic method which makes the training of Deep Neural Networks (DNN) **faster** and **more stable**.
- It consists of **normalizing activation vectors from hidden layers** using (mean and variance) of the current batch. 
- This normalization step is applied right before (or right after) the nonlinear function.

    ![image41]
    
    ![image42]

# Optimizer <a name="optimizer"></a> 

# (Batch) Gradient Descent <a name="gradient_descent"></a>
- It iterates over the **whole Training set** before updating the weights
- Each update is very small, as the learning rate is small
- GD (also called Batch-GD)is slow

    ![image43]

## Gradient Descent as optimization algorithm <a name="grad_dec"></a>
- The last ingredient is the **optimization algorithm**. The most commonly used one is the **gradient descent**. The main point is that we can find the **minimum of a loss function** by applying the rule: 𝑥𝑖+1 = 𝑥𝑖 − 𝜂𝑓′(𝑥𝑖) , where 𝜂 is a small enough positive number. In machine learning, 𝜂, is called the learning rate. The rationale is that the first derivative at xi , f’(xi) shows the slope of the function at xi.

    ![image21]
    ![image27]
    ![image28]

- Learning rate - Best Practice:

    ![image22]

    [Source: Udemy - The Data Science Course 2021: Complete Data Science Bootcamp ](https://www.udemy.com/course/the-data-science-course-complete-data-science-bootcamp/)


## Gradient Descent algorithm <a name="grad_dec_algo"></a>
- Update weights and bias for **every** point.

    ![image13]

## Forward and Backward propagation <a name="for_and_back"></a>
- Forward and backward propagation are needed for learning. 

    ![image24]

    [Source: Udemy - The Data Science Course 2021: Complete Data Science Bootcamp ](https://www.udemy.com/course/the-data-science-course-complete-data-science-bootcamp/)

## Minimal example in NumPy <a name="min_example"></a>
- Open Jupyter Notebook ```NN_numpy.ipynb```.
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

# Stochastic Gradient Descent (SGD) <a name="sgd"></a>
- Similar to GD
- But it updates weights many times in a single epoch
(in principle after each sample)
- Take care: A lot of people do not distinguish between SGD and minibatch GD.

    ![image44]

# Minibatch Gradient Descent <a name="mini_batch"></a> 
- Similar to SGD, but 
    - **SGD**: weight update after **each sample**
    - **Minibatch GD**:  weight update after **each batch**
- Batching process: Process of splitting the dataset into n batches (minibatches)
- Much faster than GD
- Small disadvantage: It approximates things a little bit --> Loss in accuracy. Example: 
    - 10000 samples
    - For a batchsize of 1000 --> 10 batches --> 10 weight updates per epoch 
- But approximation induces noise. **Noise can help to overcome local minima**.

    ![image45]
- Why is SGD (or minibatch GD) faster?
    - related to hardware
    - CPU or GPU cores train different batches simultaneously.

# Momentum <a name="momentum"></a> 
- Problem (especially) for GD 
    - **At low lr**: It often falls in a **local minimum** instead of a global one
    - **At higher lr**: It can overcome a **local minimum** but under risk of **oscillations**

    ![image46]

- Solution: Momentum

    ![image47]
    - Add earlier gradients to the actual gradient to overcome local minima.

# Learning Rate <a name="lr"></a>  
- Learning Rate **Linear Schedule**, 
    - e.g. decrease lr every 5 epochs by 10
    - Loss converges much faster
- Learning Rate **Exponential Schedule**:
    - &eta; = &eta;<sub>0</sub> e<sup>-n/c</sup>
    - n = current epoch
    - c = decay coefficient, hyperparameter
- **Adaptive Learning Rate**
    - It dynamically varies the learning rate at each update and for each weight individually.

# Hyperparameters vs. Parameters <a name="hyper_para"></a> 

| Hyperparameters (preset) | Parameters (found by optimizing) |
| --- | --- |
| width | weights (w) |
| depth | bias (b) |
| epochs ||
| lr||
| batch size| |
| momentum coefficient &alpha; ||
| decay coefficient c ||
| L1/L2 reg. weight||
| RNN: cell type (LSTM, Vanilla, GRU) ||
| RNN: depth of model, stacked LSTMs||

# AdaGrad and RMSProp <a name="adagrad"></a> 
- AdaGrad = Adaptive Gradient
- RMSProp = Root Mean Square Propagation
- Scalable learning rate

    ![image48]

    - For RMSProp we use another hyperparameter, decay rate &beta;. This hyperparameter is similar to the decay rate of momentum.

# Adam <a name="adam"></a>
- Adam combines 
    - **adaptive learning rates** with
    - **momentum**

    ![image50]

    Adam combines all:
    - **SGD**: minibatching, multiple updates during one epoch --> faster learning
    - **Adaptive LR (AdaGrad)**: 
        - if dL/dw is flat  --> increase LR
        - if dL/dw is large --> decrease LR
    - **RMSProp**: Moving scheduler --> weight updates in both directions (w<sub>i</sub> + ... or w<sub>i</sub> - ...)
    - **Momentum**: Take the momentum of previous weight steps to overcome local minima.  

# Preprocessing <a name="preprocess"></a>
It is any manipulation - **data transformation** - of the dataset before running it through the model 
# Why Preprocessing? <a name="why_preprocess"></a>
- **Compatibility**, e.g. Tensorflow works with Tensors and not with Excel sheets
- **Orders of magnitude**, e.g. a linear combination of input variables of different orders is problematic.
- **Generalization**, same model but different issue, a trained network can be used for a different topic  

# Types <a name="types"></a> 
- Relative Values
- Logarithms
- Standardization
- Normalization
- Principal Component Analysis
- Whitening
- Treating Categorical Data
- Data Augmentation

## Relative Values <a name="rel_val"></a> 
- Instead of absolute ones
- e.g. [stock prices](https://www.google.com/search?client=firefox-b-d&q=apple+stock+price) 
- Useful for Time Series Data

## Logarithms <a name="loga"></a> 
- Log transformed data 

    ![image51]

## Normalization <a name="normal"></a> 
- Normalization simply scales the values in the range [0-1]. 
- To apply it on a dataset you just have to subtract the **minimum value from each feature** and divide it with the **range (max – min)**.

    ![image52]

## Standardization <a name="standard"></a> 
- Standardization on the other hand transforms data to have a **zero mean** and **one unit standard deviation**.

    ![image53]

## Normalization with L1 or L2 norm <a name="norma_l1_l2"></a>
- The L1 norm that is calculated as the sum of the absolute values of the vector.

    ![image54]

- The L2 norm that is calculated as the square root of the sum of the squared vector values.

    ![image55]

## Principal Component Analysis <a name="pca"></a> 
- Intersting articel: [A One-Stop Shop for Principal Component Analysis](https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c)
### What is PCA?
-  **Dimension reduction** technique. Two classes
    - **Feature Elimination**: Reduce the feature space by eliminating features 
    - **Feature Extraction**: extract most important features --> PCA
    
- **PCA**:
    - Find feature axis of maximum variance in high dimensional data.
    - Project data points into this new feature space
    - New feature axis are orthogonal 
- Input: **x**-vector with **d** dimensions
- Output: **z**-vector with **k** dimensions 
- Transformation matrix: **W** with **d x k** dimensions

    ![image56]

### PCA Algorithm:
1. Provide tabular data with **n** rows, 1 column with dependent variable **Y** and **p** columns with independent variables (the matrix of which is usually denoted **X**)
2. Take the matrix of independent variables **X** and, for each column, subtract the mean of that column from each entry. (This ensures that each column has a mean of zero.)
3. Decide whether or not to standardize. Given the columns of **X**, are features with higher variance more important than features with lower variance, or is the importance of features independent of the variance? (In this case, importance means how well that feature predicts **Y**.) If the importance of features is independent of the variance of the features, then divide each observation in a column by that column’s standard deviation. (This, combined with step 2, standardizes each column of **X** to make sure each column has mean zero and standard deviation 1.) Call the centered (and possibly standardized) matrix **Z**.
4. Take the matrix **Z**, transpose it, and create **ZᵀZ**). The resulting matrix is the covariance matrix of Z, up to a constant. This matrix has the dimension **d x d** (d = dimension of variables).
5. Calculate the **eigenvectors** and their corresponding **eigenvalues** of **ZᵀZ**. Use eigendecomposition of **ZᵀZ** where we decompose **ZᵀZ** into **PDP⁻¹**, where **P** is the **matrix of eigenvectors** and **D** is the **diagonal matrix with eigenvalues on the diagonal** and values of zero everywhere else. The eigenvalues on the diagonal of D will be associated with the corresponding column in **P** — that is, the first element of **D** is **λ₁** and the corresponding eigenvector is the first column of **P**
6. Take the eigenvalues **λ₁**, **λ₂**, …, **λ<sub>p</sub>** and **sort** them from **largest to smallest**. In doing so, **sort the eigenvectors in P accordingly**. (For example, if **λ₂** is the largest eigenvalue, then take the second column of **P** and place it in the first column position). Call this **sorted matrix of eigenvectors P'**. Note that these eigenvectors are independent of one another.
7. Calculate **Z'= ZP'**. This new matrix, **Z'**, is a centered/standardized version of **X** but now each observation is a combination of the original variables, where the weights are determined by the eigenvector. As a bonus, because our eigenvectors in **P'** are independent of one another, **each column of Z' is also independent of one another**!
8. Finally, we need to determine how many features to keep versus how many to drop. Because each eigenvalue is roughly the importance of its corresponding eigenvector, the **proportion of variance explained** is the sum of the eigenvalues of the features you kept divided by the sum of the eigenvalues of all features.

    ![image57]

    Hence calculate the cumulative proportion of variance explained. For example: If you like to keep variables which explain up to 90% of the variance, done by the first ten out of fifteen variables, then keep the first ten and drop the last five.

    ![image58]


## Whitening <a name="white"></a> 
## Treating Categorical Data <a name="categorical"></a> 
Two interesting approaches:
- One-Hot-Encoding
- Binary Encoding

    ![image59]

## [Data Augmentation](#data_aug) <a name="data_aug"></a> 
   
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
