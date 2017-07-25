# Tensorflow LSTM using Nietzsche
#### Satchel Grant - July 24, 2017

### Overview
The goal of this project was to familiarize myself with the inner workings of LSTMs using TensorFlow. The idea for using Nietzshe came from [Fast.ai](http://www.fast.ai/). Jeremey Howard uses this text to demonstrate how recurrent neural networks and LSTMs work. I enjoy Nietzsche's philosophies so I was curious to see what kind of blunt, existential truths the computer could generate. Disappointingly, much of the generated text does not make gramatical sense. A fun future project would be finding a way to enforce proper grammer.

This project includes 2 different scripts that employ LSTMs. One processes the data using characters whereas the other processes the text using full words. The text was tokenized using NLTK.

### System Requirements
The following packages will ensure the use of these scripts:

- Python 3.5
- TensorFlow 1.0.0
- Numpy 1.12.0
- NLTK 3.2.4
- Keras 2.0.4

### Walk Through
Recurrent Neural Networks (RNNs) are a way to use neural networks over a sequence of data. Some data is senseless without context. A sentence, for example, is made up of a series of data points known as words. Each word holds meaning to us with or without context, but a sequence of words can encode a larger, more complex idea. The words act as building blocks for meaning and ideas. The ordering and placement of these words within the text is crucial to understanding the greater meaning. In order to extract the greater method using automation, we need methods that act over a sequence of data. RNNs are one such method.

RNNs store and update a memory of past data in a sequence. New data is received, manipulated, and then combined with the stored memory to create an updated memory. The memory can then be used as a meaningful representation of the sequence of data. We will be using this memory to generate words.

A Long Short Term Memory (LSTM) is a specific type of RNN. It uses a gating system to ensure the internal memory is updated appropriately.
