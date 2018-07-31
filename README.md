# Predictive Maintenance 

### Confirmation
After making plenty of researches I found more specific information on 
<a href="https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan">this</a>
website about the data and the problem in order to have an imagination which makes me not only understand
but also feel the data. As I understand from the article there're 4 engines data collected with 
21 sensors. So I've got 4 train, 4 test and 4 RUL data.

### My approach
My approach is to train a model with <a href="http://colah.github.io/posts/2015-08-Understanding-LSTMs/">LSTM</a> 
(Long Short Term Memory) RNN (Recurrent Neural Network) separately for 4 engines. The reason I did 
separately is if you imagine the plane with 4 engines on it wings there're not placed or attached 
in the same position so what I mean is there are placed in different coordination. In a nutshell 
my assumption is the effort each engine gets when it starts are not the same that's why I decided 
to build a model and train them separately afterwards test them separately either.

Approach looks like this:
* 1st model(train)-> (1test with 1RUL)

* 2nd model(train)-> (2test with 1RUL)

* 3rd model(train)-> (3test with 1RUL)

* 4th model(train)-> (4test with 1RUL)

### What I did?
First of all I developed the model by using PyCharm IDE however I've got a problem with performance
while testing the model. I needed to test the data with 100 epochs and maximum epoch I've
reached was 36 which is inadequate and took quit long. The result from the test was not
satisfactory so I researched Virtual Environment and I reached Colab developed by Google 
where I could test by using GPU and had satisfactory results. 

##### Colab
You can try the code directly on [Colab](https://colab.research.google.com/drive/1nyhbz_zcVF2upQqVIBIxh1gju1vxi2mr#scrollTo=edR3gkrYaR3H).
Save a copy in your drive and enjoy It!

##### Conda Environment
* Python 3.6
* [numpy 1.13.3](http://www.numpy.org/)  is the fundamental package for scientific computing with Python.
* [scipy 0.19.1](https://www.scipy.org/) is a Python-based ecosystem of open-source software for mathematics, science, and engineering. 
* [matplotlib 2.0.2](https://matplotlib.org/) For Visualization
* spyder 3.2.3
* [scikit-learn 0.19.0](http://scikit-learn.org/stable/) Simple and efficient tools for data mining and data analysis
* [h5py 2.7.0](https://www.h5py.org/) The h5py package is a Pythonic interface to the HDF5 binary data format. 
* [Pillow 4.2.1](https://pillow.readthedocs.io/en/latest/) Pillow is the friendly PIL fork by Alex Clark and Contributors. PIL is the Python Imaging Library.
* [pandas 0.20.3](http://pandas.pydata.org/) library providing high-performance, easy-to-use data structures and data analysis tools for the Python
* Anaconda 3
* [TensorFlow 1.3.0](https://www.tensorflow.org/)
* [Keras 2.1.1](https://keras.io)




