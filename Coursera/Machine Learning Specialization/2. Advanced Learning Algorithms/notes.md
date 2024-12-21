# Advanced Learning Algorithms

Second course on the Machine Learning Specialization

## Intro to Neural Networks

They were originally tried to be model after the human brain in the 1950s but have considerably evolved more towards engineering principles, not really similar anymore to anything biological since the brain is too complicated. With major breakthroughs at the start of the century starting with speech recognition, computer vision, and more recently NLP.

A neuron can have inputs and outputs.

![[Pasted image 20240322212117.png]]

As time has gone by and with inventions like the Internet, we now have more data and NNs really seem to perform better with the more data we have. NNs, benefit from Big Data.

### Intuition

A neural network consists of multiple layers, where each layer has one or multiple neurons. Each neuron, in practice, receives inputs from every neuron in its previous layer, and sends an output to every neuron in the following layer.

![[Pasted image 20240324150244.png]]

Every input to a neuron can be thought of as a feature. The computation inside a single neuron is like an independent logistic regression, which passes its inputs (or outputs from previous layer $\vec{x}$ or $\vec{a}$, the parameters $W$ and $\vec{b}$. We pass the result $\vec{z}$ through an activation function $a(z)$ to obtain the layer's output or **activation values** ($\vec{a}$) or probabilities, to be passed as features to the following layer. 

- The vector of all activation values of a layer is $\vec{a}$. 
- The matrix of weights assigned to a layer, where every row are the weights of each connection to a unit in the following layer, can be defined as $W$.

The entire neural network thus can be seen as a way to try to obtain better and better features for the following layer. As we pass through hidden layers, we should be getting more representative features layer by layer. In the end, on our last computation, on the output layer, we should have the best and most representative features we can possibly have for a final logistic regression. So, it is like if the NN can learn its own features.

[Cool explanation of logistic regression with NNs](https://stats.stackexchange.com/a/500973)

![[Pasted image 20240322220730.png]]

Basic questions of Neural Network Architecture:

- How many units (neurons)?
- How many layers?

![[Pasted image 20240322222943.png]]

A practical example of a neural network. If we had a grid of 1000 x 1000 pixels we would have 1 million input features, each being a pixel, and insert it into the neural network as a column vector of pixels in consequent rows.

What would happen through each layer of the neural network would be that we start to find patterns, such as outlines made of pixels, and then part of faces, and then complete faces, and subsequently until we output a probability for each face we have saved.

### Layers

Each unit in a layer receives a weight vector and a bias, and outputs an activation value. The set of all activation values of a layer is assigned to each unit in the following layer.

The notation for layers is put as a superscript and the specific unit within that layer is denoted with a subscript, so $\vec{w}^{[1]}_2$ for the first layer, second unit's weights.

For example:

$a^{[3]}_2=g(\vec{w}^{[3]}_2\cdot \vec{a}^{[2]} + b^{[3]}_2)$

> The activation value of the 2nd unit at the 3rd layer is equal to the output of the activation function, passed in the unit's bias and vector of weights dot product with all the activation values from the previous layer (2nd layer).

So, the general formula for the activation value of a unit is:

$a^{[l]}_j=g(\vec{w}^{[l]}_j\cdot\vec{a}^{[l-1]}+b^{[l]}_j)$, where for the first layer the inputs act as the activation values: $\vec{x}=\vec{a}^{[0]}$.

## Forward Propagation

Called that way because it consists of propagating activation values in the forward direction, from left to right, through the neural network. It is the process in which, by having a specific set of parameters and biases already stablished by someone else (like when we used a pre-trained model), we can pass in our data and infer an output. In the end, we can decide a threshold to interpret our output activation value.

A common neural network architecture technique is to create layers with decreasing units since it appears to work better. [More on how this works](https://www.quora.com/Why-is-it-common-in-Neural-Network-to-have-a-decreasing-number-of-neurons-as-the-Network-becomes-deeper):

> *$\ldots$ this can help the network compress the data and focus on important features. The goal is to learn a compact representation of the data that captures the essential information while discarding less relevant details. The network learns to represent the data efficiently during training.*
>
> - Gemini

![](assets/Pasted%20image%2020240415001819.png)
#### Lab: Intro to Tensorflow 

- Keras was created independently by a single individual which works on top of Tensorflow and is more simple and layer-centric.
- A tensor is just another name for an n sized array

```python
import tensorflow as tf
# Create an example layer of 1 unit and linear act. func.
# also known as the "no activation" or "identity function"
linear_layer = tf.keras.layers.Dense(
	units=1, 
	activation='linear', # we can also use 'sigmoid'
	name='Layer1',
	input_dim=1 # n features, model will receive (*,1) shapes
)
# Create a layer for a new binary classification model
logistic_layer = tf.keras.layers.Dense(1, input_dim=1,  activation = 'sigmoid', name='L1')
# Set and get weights
layer.set_weights([[10.]], [1.]) # w is (m,n), b is (m)
layer.get_weights()
# Add the layer to a sequential model
model = tf.keras.Sequential()
model.add(logistic_layer)
model.get_layer('L1')
# Get summary of model, returning layer names, etc.
model.summary()
# Predict with model, perform forward propagation
a1 = model.predict(X) # X is (m,n), a1 is (units,labels)
```

## Tensorflow

```python
# Single training example
x = np.array([[200., 30.]])
# Create layer 1 and get activation values
# (when weights are not set, they are random)
layer_1 = tf.keras.layers.Dense(3, activation='sigmoid')
a1 = layer_1(x)
# Create layer 2 and get activation values
layer_2 = tf.keras.layers.Dense(1, activation='sigmoid')
# We are passing the activation values from the previous layer
a2 = layer_2(a1)
```

The `Tensor` data type is a class created inside of Tensorflow with the objective of performing calculations between matrices more efficiently, but in reality it could be interpreted like an ndarray. Because of how matrix multiplication is defined inside, we always need to provide 2D arrays, even for 1 x m matrices, done for better computations

`Tensor.numpy()` to get a numpy array from the tensor.

A sequential model is basically just concatenating together layers so that we can perform propagation between layers more easily, by just specifying the input for the first layer and obtaining an array from the last one.

```python
model = tensorflow.keras.Sequential([layer_1, layer_2])
model.compile(...)
# Train the model
model.fit(X_train, y_train)
# Forward propagation
y_test = model.predict(X_test)
```

#### Lab: Forward Propagation with Tensorflow

When we want to train a model we might want to duplicate our dataset into new $2*m$ training examples. The reason why we can copy our sample multiple times, thus increasing the number of training examples (just the original examples but multiple times), is to have less epochs in the training phase. An epoch is a complete run over the training examples. In TF, with each epoch there is an overhead (excess computational resource usage / runtime) which can be avoided doing this. [In-depth explanation](https://community.deeplearning.ai/t/what-is-the-meaning-of-reduce-the-number-of-training-epochs/386807/2)

The reason of the overhead can be explained by different factors, such as repeatedly saving information for later visualization of the model training process or others, like saving the state of the model at every epoch completion.

Doing the above with 200 samples copied 1000 times, gives us a final number of 200,000 training examples. Since after each epoch we record information about the training process (like the changing loss) and also define early stopping conditions, we would like to have a reasonable amount of epochs while also decreasing its overhead. What we would get with 10,000 epochs, we get now with 10 epochs.

```python
# Set seed for random number generations (for consistensy and repeatability)
tf.random.set_seed(1234)
# Create model (optionally set input layer size)
# Parameters, not specified, are created at random
model = Sequential([
	Dense(3, activation='sigmoid', name = 'layer1'),
	Dense(1, activation='sigmoid', name = 'layer2')
])
# Define loss function and optimizer for training
model.compile(
	loss = tf.keras.losses.BinaryCrossentropy(),
	optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
)
# Train the model
mode.fit(X, y, epochs=10)
```

Through each layer, we could consider that each unit "contributes" to the result on a unit on the following layer:

![[Pasted image 20240329200020.png]]

### Python Implementation

![](assets/Pasted%20image%2020240405010112.png)

The basic Python implementation involves doing all of the calculations only using Numpy. The flow of these calculations to perform forward propagation is:

1. Define the weight vector and bias which corresponds with a unit.
2. Define the logit function and its activation with the sigmoid function
3. Do the above steps for every unit, thus obtaining a weight matrix and bias vector for a layer
4. Do the above for every layer, thus we would have multiple weight matrices and bias vectors
5. Order the layers sequentially, so that the previous output is the input of the next
6. Run all the training examples through the sequence

## Additional: More vectorization

$z=\vec{a}\cdot\vec{w}$   is equivalent to    $z=\vec{a}^T\vec{w}$.

This means that we can compute a dot product using matrix multiplication.

A good tip: If we see a **transpose** of a matrix, let's keep our focus on its **rows**, if we see a normal **matrix**, we look at the number of **columns** it has.

![](assets/Pasted%20image%2020240515003154.png)

![](assets/Pasted%20image%2020240515003416.png)

### Lab: NNs recap with Tensorflow

Very interesting lab to make a quick recap of the entire process [here](https://github.com/greyhatguy007/Machine-Learning-Specialization-Coursera/blob/main/C2%20-%20Advanced%20Learning%20Algorithms/week1/C2W1A1/C2_W1_Assignment.ipynb)

There's also an explanation on Numpy broadcasting

![](assets/Pasted%20image%2020240515225711.png)

## Activation Functions

So again, why can't we just use the linear activation? 

Because we can't add non-linearity by just using linear functions. Because concatenating a linear function to a linear function will always yield another linear function. Same thing with logistic regression when only the output layer is sigmoid, we would yield a regular logistic regression.

![](assets/Pasted%20image%2020240515232658.png)

Activation functions can make your model more powerful depending on your specific case, what your data models, what kind of relationships there are between them, etc. For example, how many people are aware of a product could be something we might want to quantify further than just from 0 to 1, for example, when something goes viral on social media it can have gigantic awareness.

### ReLU

For this last example we might want an activation function which allows for big values but not for negative values, like:

$$
\begin{align}
\text{ReLU:}\ \ g(z)=max(0,z)
\end{align}
$$

But how do we decide between them?

- Output layer: 
	- Classification: sigmoid
	- Regression: linear (none) or ReLU if output can only be positive.
- Hidden layers: 
	- The industry has changed from using sigmoid to ReLU by far.
	- Why? The sigmoid function has two areas where it flattens while ReLU only has one. This means that [saturation](4TO%20SEMESTRE/JUMPSTART/Cloud%20Skill%20Boost/2.%20Launching%20into%20ML.md#Advanced%20Logistic%20Regression) won't happen as much. Plus, its faster to calculate.
- Don't use linear activation in hidden layers, as it prevents the network from learning complex patterns by only performing linear transformations, effectively reducing the network to a single linear transformation.

[Great explanation of why ReLU is better for hidden layers and intuition non-linearity](https://www.linkedin.com/pulse/rectified-linear-unit-non-linear-mukesh-manral/)

### Softmax

To solve multi-class classification problems we use the SoftMax function:

$$
\begin{align}
z_j=\vec{w}_j\cdot x+b_j,\ \ \ j=1,\ldots,N\\
\\
a_j=\frac{e^{z_j}}{\sum^N_{k=1}e^{z_k}}=P(y=j|\vec{x})\\
\\
a_1+a_2+a_3+\ldots+a_n=1\\
\end{align}
$$
But what does this mean?

- There are $j$  classes, which means the output layer would have $j$ units
- Each unit has its own vector of weights $\vec{w}$ and bias $b$ because of connections with the units in the previous layer
- We calculate each logit $z_j$ for each class with these weights and biases
- We calculate each activation value $a_j$ as the probability that the label is that class $j$. The probability is obtained as simple as doing:

$$
\begin{align}
\frac{\text{this class value}}{\text{sum of all classes' values}}=\frac{v}{\sum^N_{k=1}v_k}=\frac{e^{z_j}}{\sum^N_{k=1}e^{z_k}}
\end{align}
$$

- All probabilities must sum up to 1, which is 100%

![](assets/Pasted%20image%2020240516001620.png)

In Tensorflow, SoftMax has the name `SparseCategoricalCrossentropy` and as other loss functions is located in `tf.keras.losses`.

One thing we must always consider are numerical roundoff errors, which show up because floating point numbers are saved in limited memory space. The two following equations are the same, but the one on the right is more accurate and has less roundoff errors because of how Tensorflow implements and calculates equations.

$$
\begin{align}
\frac{1}{1+e^{-z}}=\frac{1}{1+e^{-(\vec{w}\cdot \vec{x}+b)}}
\end{align}
$$

To correct this in Tensorflow:

```python
# Regular, basic implementations, with more roundoff errors:
model = Sequential([
	Dense(units=25, activation='relu'),
	Dense(units=15, activation='relu'),
	Dense(units=10, activation='softmax'), # We leave this as softmax
])
model.compile(loss=SparseCategoricalCrossentropy())
```

```python
# More accurate, less roundoff errors:
model = Sequential([
	Dense(units=25, activation='relu'),
	Dense(units=15, activation='relu'),
	Dense(units=10, activation='linear'), # We change this to linear
])
# Since the output layer outputs the raw data (the logits) because of the no activation function in it, we specify that the output are logits to use them effectively
model.compile(loss=SparseCategoricalCrossentropy(from_logits=True))
```

Final 10-digit MNIST neural network:

![](assets/Pasted%20image%2020240516005601.png)

When performing multiclass classification, we know that every unit is "mapped" to one class. To be precise, the activation value of each of these units must have its greatest value when the real class is that "mapped" class. So, if we have an example with a real class of `3`, then the unit that is mapped to that class, say $a_3$, must have the greatest probability (activation value) from all other units.

## Advanced Optimization

The Adam (for Adaptive Moment) optimizer improves the process to find the minimum loss by changing the learning rate in an adaptive manner. It is based on the RMSP and Momentum algorithms, getting the best of both. 

It has become a standard optimizer in the industry over many, including simple SGD.

The basic intuition behind the algorithm is:

![](assets/Pasted%20image%2020240529184342.png)

