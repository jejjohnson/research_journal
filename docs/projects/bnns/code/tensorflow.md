# TensorFlow

TensorFlow is by far the most popular deep learning framework to date. It is the preferred choice in production and it is also still widely used in research. TensorFlow itself is actually written in C++ (and CUDA) but there are various APIs that allow you to call those functions with other languages like Python, Swift, Go and JavaScript. 

---

## From TF 1.X to TF 2.X

If you are already familiar with TF1.X, there are some key things you need to know about what has changed since TF2.X. The original TensorFlow (TF1.X) wasn't wasn't very pythonic even though the API was in Python. Among other [nasty sharp edges](https://jacobbuckman.com/2018-09-17-tensorflow-the-confusing-parts-2/), you had to define the graphs statically which required you to compile your graph before running it. This took us back to the C++ days where we have to compile-run-modify-repeat. In addition if something went wrong, we didn't have the training to be able to decipher the error messages. So on the plus side, if you could gain control of TF back then, you had to know what you were doing. It wasn't so easy. But on the other hand, most people programming were not computer scientists so naturally their code was very messy and very difficult to read. There wasn't a very good standard and so "reproducible code" was a nightmare to go through; *very similar to "reproducible MATLAB" code because people do not tend to follow any set standard...except maybe spaghetti*. Later they added eager execution which allowed you to define parts of your graph dynamically and then run them as you add more parts without needing to compile it. This was much better and it became easier to use TF without needing to worry about graphs. 

In tandem, a library called keras was gaining popularity. This library was basically a wrapper to hide all of the 'boilerplate code' so that a different class of users (beginners) can get started without needing to be bothered with the details. TF2.X is a more

### Model Building

There is an example below to demonstrate the readability aspect for defining a simple linear regression model. 

<!-- tabs:start -->

#### ** TF 1.X - Static **

```python
# Create Graph
lr_graph = tf.Graph()

with lr_graph.as_default():
    x = tf.placeholder(name="x", dtype=tf.float)
    y = tf.placeholder(name="y", dtype=tf.float)

    # WEIGHTS
    w_init = tf.random_normal_initializer()
    w = tf.Variable(
        initial_value=w_init(shape=(units,), dtype=tf.float32),
        name="w"
    )
    # BIAS
    b = tf.Variable(
        initial_value=w_init(shape=(units,), dtype=tf.float32),
        name="b"
    )
    # MODEL OPERATIONS
    y_hat = tf.add(tf.matmul(x, w), b)
    loss = tf.reduce_mean(tf.squared(y - y_hat))

    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)
```

#### ** TF 2.X - Dynamic **

```python
class LinearRegression(tf.keras.model):
    def __init__(self, units=32, input_dim=32):
        super().__init__()
        # WEIGHTS
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype='float32'),
            trainable=True
        )
        # BIAS
        self.b = tf.Variable(
            initial_value=w_init(shape=(units,), dtype='float32'),
            trainable=True
        )
    
    def call(self, x):
        return tf.matmul(inputs, self.w) + self.b
```
<!-- tabs:end -->

**Source**: [Medium Post](https://medium.com/red-buffer/tensorflow-1-0-to-tensorflow-2-0-coding-changes-636b49a604b)

Notice that the main difference is that the default TF1.X has to define all of the operations as a graph before doing anything else. Whereas TF2.X, there is no need to do that. In addition, the keras API is closely linked to the TF library so they encourage you to use the standard model creation as shown by the documentation.

### Model Training

Training was a different story. For the original TF1.X you had to create a session and then all of the gradients optimization had to go through that session. It was a pain because it was necessary for everything but it basically had to follow through the code if there were any crazy training procedures. I've seen many cases where there are wild and rogue sessions that I have to keep track of in order to follow what the users were doing. In TF2.X, there are no sessions. Just a `gradientTape` which tracks the final outputs and the final weights. And if you want the gradients, ask for them. That's it. Simple. 

I have included a minimal training example to showcase the major difference between TF1.X and TF2.X.

<!-- tabs:start -->

#### ** TF 1.X - Static **

```python
# Create session
with tf.Session(graph=lr_graph) as sess:

    # Initialize all variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training Loop
    for iepoch in range(epochs):
        
        # Run optimization in session 
        feed_dict = {x: Xtrain, y: ytrain}
        sess.run(
            optimizer,
            feed_dict=feed_dict
        )

        # get losses
        loss, acc = sess.run(
            [loss_func, accuracy_func],
            feed_dict=feed_dict
        )
```

#### ** TF 2.X - Dynamic **

```python
for iepoch in range(epochs):
    with tf.GradientTape() as tape:

        # Forward pass
        ypreds = lr_model(Xtrain)

        # Loss 
        loss = loss_func(y, ypreds)

    # compute gradients
    gradients = tape.gradient(loss, lr_model.trainable_weights)

    # update weights of linear layer 
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    # update accuracy 
    acc_func.update_state(y, ypreds)
```
<!-- tabs:end -->

**Source:** [Easy TensorFlow](https://github.com/easy-tensorflow/easy-tensorflow/blob/master/2_Linear_Classifier/Tutorials/1_Linear_Classifier.ipynb)

There are some more changes which you can read [here](https://www.tensorflow.org/guide/effective_tf2). 

### My Final Thoughts

The biggest change I would say is the code standard and the clear spectrum of user cases. They are really promoting the `keras` way of defining `Layers`, `Models`, etc but you could also use `Sequential` or `Functional`. The rules are not absolute but this does set a nice "standard way to do things". This is a **good thing**. It's pythonic and readable. Some people are researchers and scientists whereas other people are computer scientists. But typically we'll be reading ML peoples code so we need a standard. That is if we plan on being a community and sharing. So I don't think we should spend time fighting DL libraries and reading sloppy code. We can spend more time devloping and solving more problems. One could argue that all of the details are hidden now and this promotes people just using stuff without understanding. But they're actually just optional to see. You can code from scratch if you want to. And yes, there will be many cases of people using models that they don't understand. But that's a choice and there will be barriers to prevent those people from thriving in the community. I personally think the changes are good and we shall see what the future holds for TF and DL software in general.

---
### My Favourite Resources

These are my favourite resources. I've gone through almost all of them and I found that they did the best at explaining how to use TensorFlow. 

!> **Remember**, I am coming at this from a **researchers** perspective. So I am biased and the resources I've chosen assumes some prior knowledge about **Python** programming and **Deep learning** in general. I will list some resources

---
#### Francois Chollet

The best resource I have found is from the founder of keras ([Francois Chollet](https://fchollet.com/)). He is a very outspoken individual who is very proud of keras and how it has changed the community. He also likes to make comparisons between frameworks but overall he is very passionate about his work. He is also super active on [twitter](https://twitter.com/fchollet?lang=en) and has some interesting opinions from time to time.

The first tutorial is basically a notebook on using `tf.keras` from a deep learning perspective. I think he breaks it down quite nicely and goes through all of the important aspects that a researcher should know when using TensorFlow. If you are already familiar with TensorFlow I think you'll find almost every major point he makes useful (e.g. `Callbacks`, ) when you construct your neural networks. If you still don't see it after going through it, don't worry, it will come up.

* TensorFlow 2.0 + Keras Overview for Deep Learning Researchers - [Colab Notebook](https://colab.research.google.com/drive/1UCJt8EYjlzCs1H1d1X0iDGYJsHKwu-NO#scrollTo=zoDjozMFREDU)
* `tf.keras` for Researchers: Crash Course - [Colab Notebook](https://colab.research.google.com/drive/17u-pRZJnKN0gO5XZmq8n5A2bKGrfKEUg) 
* Inside TensorFlow: `tf.keras` - [Video 1](https://youtu.be/UYRBHFAvLSs) | [Video 2](https://www.youtube.com/watch?v=uhzGTijaw8A)


---
#### TensorFlow Website

There are a few really good tutorials on the TF website that give a really good overview of changes from TF 1.X TF 2.X as well as some more in-depth tutorials. I found the tutorials a bit difficult to navigate as there is a lot of repetition with the 'nuggets of wisdom' scattered everywhere. In addition, I find that the organization isn't really done based on the users level of knowledge. So I tried to outline how I think one should approach the tutorials based on three criteria: **absolute beginner**, **Keras users**, and **PyTorch users** which is my way of saying **Beginner**, **Intermediate** and **Advanced**.

---
**1 Absolute Beginners**

Honestly, if you're just starting out with deep learning then you should probably just dive into it using a project or take your time and go through a course and/or book. I've listed my favourite books in the next section if you're interested but I will recommend this course which is sponsored by TensorFlow. I went through the first few lectures but I got a bit bored because I had already learned this stuff. But I like the balance of explanations and code.

* Introduction to TensorFlow for Deep Learning - [Udacity](https://www.udacity.com/course/intro-to-tensorflow-for-deep-learning--ud187)

If you have a bit more free time and dedication, I would recommend you go through the TensorFlow curriculum. They break the parts necessary for learning Deep learning with TF as your platform. It has books and video courses and I personally think it is organized very well. The course I listed above is included in the curriculum.

* TensorFlow Curriculums - [Learn ML](https://www.tensorflow.org/resources/learn-ml)

---
**2 Keras Users** (Intermediate)

Most people who apply DL models to data will be in this category. They will either want simple models with fairly straightforward setups or highly complex networks. In addition, they also may have complex training regimes. They all fall into this category and from this section, you should be able to get started.

* [Keras Overview](https://www.tensorflow.org/guide/keras/overview)
  > This is a fairly long tutorial that goes through keras from top to bottom. I wouldn't recommend reading the whole thing in one go as it is a bit overwhelming. If you want to do simple models, then only look at [part 1](https://www.tensorflow.org/guide/keras/overview#build_a_simple_model). And then for a quick overview of complex models, check out [part 2](https://www.tensorflow.org/guide/keras/overview#build_complex_models) 
* [Keras Functional API](https://www.tensorflow.org/guide/keras/functional)
  > I imagine most people who are experimenting with complex models with inputs and outputs in various locations will be here.
* [Train and Evaluate with Keras](https://www.tensorflow.org/guide/keras/train_and_evaluate)
  > This is another long guide that goes through how one can train your DL model. If you're not interested in too much training customization, then you'll probably mostly interest in [part 1](https://www.tensorflow.org/guide/keras/train_and_evaluate#part_i_using_build-in_training_evaluation_loops) where you use the built-in training module.  [Part 2](https://www.tensorflow.org/guide/keras/train_and_evaluate#part_ii_writing_your_own_training_evaluation_loops_from_scratch) does things from scratch.

!> **MATLAB Users**: Although I recommend you start in the absolute beginner section to get accustomed to Python, you will probably fall into this category as well. The `Sequential` API is very similar to the new DL toolbox in the latest versions of MATLAB. Unfortunately there is no GUI yet though...

---
**3 PyTorch Users** (Advanced)

All my PyTorch and advanced Python users (me included) start here. You should feel right at home with TensorFlow using the *subclassing*. The distinction between `Layer` and `Model` is quite blurry but it's similar to the PyTorch `nn.Module`. In the end, it's super Pythonic so we should feel right at home. Finally...

* [TF for Experts](https://www.tensorflow.org/tutorials/quickstart/advanced)
  > The is a 2-minute introduction to the language. If you are familiar with PyTorch then you will find this very familiar territory. It really highlights how the two packages converged.
* [Writing Custom Layers and Models with Keras](https://www.tensorflow.org/guide/keras/custom_layers_and_models)
  > This next tutorial will go into more detail about the *subclassing* and how to build layers from scratch. It's very similar to PyTorch but there are a few more nuggets and subtleties that are unique to TensorFlow.
* [Train and Evaluate with Keras](https://www.tensorflow.org/guide/keras/train_and_evaluate)
  > This is another long guide that goes through how one can train your DL model. Pay special attention to [part 2](https://www.tensorflow.org/guide/keras/train_and_evaluate#part_ii_writing_your_own_training_evaluation_loops_from_scratch) where you build things from scratch as this is most similar to the PyTorch methods.
  

---
### Books

This is a bit old school but there have recently been a lot of good books released in the past 2 years that do a very good job at teaching you Machine learning (including Deep learning) from a programming perspective. They don't skip out on all of the theory but you won't find al of the derivations necessary to write the algorithms from scratch. A nice thing is that most of the books below have code on their github accounts that you can download and run yourself. 

* Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow - Aurelien Geron (2019) - [Book](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646) | [Github]()
  > This is a best selling book and I found it to be the best resources for getting a really good overview of ML with Python in general as well as a really extensive section on TF2.0 and keras.
* Deep Learning with Python - Francois Chollet (2018) - [Book](https://www.amazon.com/Deep-Learning-Python-Francois-Chollet/dp/1617294438)
  > By the creator of Keras himself. It's a great book that goes step-by-step with lots of examples and lots of explanations.
* Python Machine Learning: Machine Learning and Deep Learning with Python, scikit-learn, and TensorFlow 2 - Raschka & Mirjalili (2019) - [Book](https://www.amazon.com/Python-Machine-Learning-scikit-learn-TensorFlow/dp/1789955750/ref=sr_1_1?keywords=Python+Machine+learning&qid=1579273871&s=books&sr=1-1) | [Github]()
  > Another great book that talks about DL as well as ML in general. 


