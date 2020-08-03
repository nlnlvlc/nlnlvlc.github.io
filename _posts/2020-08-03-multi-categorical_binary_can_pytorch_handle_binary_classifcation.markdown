---
layout: post
title:      "Multi-Categorical Binary: Can PyTorch Handle Binary Classifcation?"
date:       2020-08-03 11:25:43 +0000
permalink:  multi-categorical_binary_can_pytorch_handle_binary_classifcation
---


In my most recent project working on a [binary Image Classifcation model](https://github.com/nlnlvlc/melanoma_classification), I wanted to explore how two separate libraries would handle the training and modeling of identifying benign v. malignant melanoma. After my previous experience with Keras where I built a similar model which [identified Pneumonia in chest x-rays](https://github.com/nlnlvlc/dsc-mod-4-project-v2-1-onl01-dtsc-ft-012120), I wanted to explore image classification further. The problem, however, was that Keras demanded a lot of memory as model ran and as the machine attempted to solve more complex problems. Though I eventually took advantage of the GPU and High-Ram options from Google Colab, I wanted to explore more RAM-friendly options, looking to PyTorch to build and train my models.

To eleviate the stress on your cpu/gpu's memory, you have to manually write out all of your models (even the pre-trained models like AlexNet or ResNet) compared to Keras where you can simply write out your layers, compile, then fit with just a few lines of code. 

#### Writing a Keras Model

```
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential()
model.add(layers.Dense(64, kernel_initializer='uniform', input_shape=(10,)))
model.add(layers.Activation('softmax'))

loss_fn = keras.losses.SparseCategoricalCrossentropy()
model.compile(loss=loss_fn, optimizer='adam')
All built-in loss functions may also be passed via their string identifier:

# pass optimizer by name: default parameters will be used
model.compile(loss='sparse_categorical_crossenropy', optimizer='adam')
Loss functions are typically created by instantiating a loss class (e.g. keras.losses.SparseCategoricalCrossentropy). All losses are also provided as function handles (e.g. keras.losses.sparse_categorical_crossentropy).

Using classes enables you to pass configuration arguments at instantiation time, e.g.:

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

 #### Writing a PyTorch Model
 
 ```
 class melanomaClassifier(nn.Module):
    def __init__(self):
        super(melanomaClassifier, self).__init__()
        self.block1 = self.conv_block(c_in=3, c_out=256, dropout=0.1, kernel_size=5, stride=1, padding=2)
        self.block2 = self.conv_block(c_in=256, c_out=128, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.block3 = self.conv_block(c_in=128, c_out=64, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.lastcnn = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=56, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.maxpool(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.maxpool(x)
        x = self.lastcnn(x)
        return x
    def conv_block(self, c_in, c_out, dropout,  **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )
        return seq_block
				
# Define model and Loss
				
model = melanomaClassifier()
model.to(device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
 ```
 
As you can see, we can run a Keras model in 14 lines while in PyTorch, we did little more than define the layers of our model in 29. 

On the bright side, PyTorch's complexity requires a solid understanding of how models work and forces an architect to really dive deep to get it right. However, the added complexity in setting up a model in Pytorch makes it incredibly difficult to find out what happened when something goes wrong, where, and why.

## Using Multi-Categorical Losses for Binary Problems
Keras and Pytorch maintain many of the same Loss functions. Setting up a model to test a binary problem in Keras requires little more than ending in Sigmoid activation function, producing a single output in your last layer, and setting our loss to "binary_crossentropy" when we compile. Of course you can also produce a multi-categorical model to test the probability of both of your classes occuring and accurately classify testing images. With PyTorch, not so much. If you were to search for binary classification models using PyTorch right now, you'll notice that very few of the models, if any, are actually use Binary functions to solve binary classifications.

PyTorch has two primary binary loss functions: BinaryCrossEntropyLoss (BCELoss) and BinaryCrossentropyWithLogitsLoss (BCEWithLogitsLoss). In Keras, logits can simply be passed to a loss function when compiling. With PyTorch, you need to know whether or not your loss function requires you to specify a specific `sigmoid ` function in your layers, pass your predictions through a `sigmoid` function at the end of each prediction, or whether or not model will run simply because your tensor is the wrong type. Not only is this difficult because models can get complicated and it's easy to lose track, it's difficult because, so far, PyTorch is designed that way and far too young to get a lot of help.

If you want to do binary classifications in PyTorch, your best bet is to use CrossEntropyLoss and pass your predictions through torch.LogSoftmax(). According to nearly every model searchable, this is the most effecient way to go. You can do something similar in Keras, using a Softmax activation and categorical_crossentropy. In theory, this should give two outputs instead of one, looking at both classes and predicting the occurance of each of your classes instead of simply saying "This belongs to 1 class." For many, it forks flawlessly.

When I was working on my model, I ran into one of the pitfalls of using PyTorch's CrossEntropyLoss for a Binary problems: it only tested and trained a single class. If you take a look at [the notebook for the first PyTorch model](https://github.com/nlnlvlc/melanoma_classification/blob/master/pyTorch%20Model.ipynb) you might notice something off about the loss and accuracy. As the epochs continue, the loss doesn't seem to improve by much while the accuracy constantly jumps between the same high and the same low points. Good news, the models can correctly classify every single malignant image as malignant. The bad news? The model classified every single benign image as malignant as well. And I wasn't the only one to experience my model failing to train and test for a particular class - sort of.

Remember, CrossEntropy and LogSoftmax are multi-categorical loss functions so most examples of this happening involve more than 2 classes. In the cases of most people asking for help, one particular class failed to train properly while the rest, often another 5+ classes, did well. The solutions followed given by helpful experts (actually, just one who answered nearly every question on the PyTorch forum) were specific to multiple classes, making it almost impossible to apply to a scenario where half of your classes are being ignored.

Ideally, the simple solution would be to switch to one of the BCE and sigmoid loss functions, but that brought on another set of challenges.

1. Unlike Keras, you can't simply swap out your output, loss, and activation functions.

2. With PyTorch, you have to be mindful that Binary functions cannot process LongTensors, only FloatTensors. And while FloatTensors are the default tensor throughout the entire model, switching to a binary function can actually change that.

3. Casting LongTensors to FloatTensors might actually break your model, if it can even be done, at all.

Unfortunately, this was a common problem that popped up earlier this year. And while it does have some work arounds for the lucky few, for many it can be unsolvable - unless you change to a CrossEntropyLoss. But what happens when your model breaks with categorical AND binary loss functions?

Who knows?

## Conclusion: Keras or PyTorch
I've seen the magic that is PyTorch and how it can outperform Keras in accuracy, sometimes in much less time and without the memory load. However, when it comes to binary functions and problems, it leaves a lot to be desired.

This isn't to say that PyTorch is bad. In fact, I'll continue to learn and use it in the future. But it's relatively new. And while it's not much younger that Keras, the complexity it takes to build a binary model coupled with the errors that are typically solvable only by making a non-binary model does highlight that the easy of setting up and troubleshooting a Keras model outweighs the trade-offs like memory heavy processes. PyTorch is great for large, multi-categorical process, whether trained manually or using a pretrained model, but it has a long way to go before it becomes optimal for Binary processes using Binary functions
