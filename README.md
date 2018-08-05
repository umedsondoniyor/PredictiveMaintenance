# Predictive Maintenance 

### Confirmation
After making plenty of researches I found more specific information on 
<a href="https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan">this</a>
website about the data and the problem in order to have an imagination which makes me not only understand
but also feel the data. As I understand from the article there're 4 engines data collected with 
21 sensors. So I've got 4 train, 4 test and 4 RUL data. 
The network uses simulated aircraft sensor values to predict when an aircraft engine will fail in the future so that maintenance can be planned in advance.
The question is "Given these aircraft engine operation and failure events history, can we predict when an in-service engine will fail?"

### My assumption
To be more specific my assumption on this project is the result may differ for the same engine if it's used in different airport or destination because the frequency of landing and taking off the plane with different destination changes for every flight. 

### My approach
My approach is solving a task with machine learning model:
  	
    * Regression models: How many more cycles an in-service engine will last before it fails?

To train a model I used <a href="http://colah.github.io/posts/2015-08-Understanding-LSTMs/">LSTM</a> 
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
Save a copy in your drive and enjoy it!

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


#### Results of Regression model
__1st Engine Result done with 59 epoches__

Train on 14849 samples, validate on 782 samples

|Mean Absolute Error|Coefficient of Determination (R^2)|
|----|----|
| 13.498672733516765|0.8166509073333222|

The following pictures shows the trend of loss Function, Mean Absolute Error, R^2 and actual data compared to predicted data: 
<p align="center">
  <img src="https://github.com/umedsondoniyor/PredictiveMaintenance/blob/master/Output/colab/engine1/model_loss.png"/>
</p>
<p align="center">
  <img src="https://github.com/umedsondoniyor/PredictiveMaintenance/blob/master/Output/colab/engine1/model_MAE.png"/>
</p>
<p align="center">
  <img src="https://github.com/umedsondoniyor/PredictiveMaintenance/blob/master/Output/colab/engine1/model_r2.png"/>
</p>
<p align="center">
  <img src="https://github.com/umedsondoniyor/PredictiveMaintenance/blob/master/Output/colab/engine1/prediction.png"/>
</p>


__2nd Engine Result done with 48 epoches__

Train on 38721 samples, validate on 2038 samples

|Mean Absolute Error|Coefficient of Determination (R^2)|
|----|----|
| 12.640330181712857|0.8787733749326216|

The following pictures shows the trend of loss Function, Mean Absolute Error, R^2 and actual data compared to predicted data: 
<p align="center">
  <img src="https://github.com/umedsondoniyor/PredictiveMaintenance/blob/master/Output/colab/engine2/model_loss.png"/> 
</p>
<p align="center">
  <img src="https://github.com/umedsondoniyor/PredictiveMaintenance/blob/master/Output/colab/engine2/model_MAE.png"/>
</p>
<p align="center">
  <img src="https://github.com/umedsondoniyor/PredictiveMaintenance/blob/master/Output/colab/engine2/model_r2.png"/>
</p>
<p align="center">
  <img src="https://github.com/umedsondoniyor/PredictiveMaintenance/blob/master/Output/colab/engine2/prediction.png"/>
</p>


__3rd Engine Result done with 67 epoches__

Train on 18734 samples, validate on 986 samples

|Mean Absolute Error|Coefficient of Determination (R^2)|
|----|----|
| 21.350444003486245|0.6057153666962475|

The following pictures shows the trend of loss Function, Mean Absolute Error, R^2 and actual data compared to predicted data: 
<p align="center">
  <img src="https://github.com/umedsondoniyor/PredictiveMaintenance/blob/master/Output/colab/engine3/model_loss.png"/>
</p>
<p align="center">
  <img src="https://github.com/umedsondoniyor/PredictiveMaintenance/blob/master/Output/colab/engine3/model_MAE.png"/>
</p>
<p align="center">
  <img src="https://github.com/umedsondoniyor/PredictiveMaintenance/blob/master/Output/colab/engine3/model_r2.png"/>
</p>
<p align="center">
  <img src="https://github.com/umedsondoniyor/PredictiveMaintenance/blob/master/Output/colab/engine3/prediction.png"/>
</p>



__4th Engine Result done with 87 epoches__

Train on 46359 samples, validate on 2440 samples

|Mean Absolute Error|Coefficient of Determination (R^2)|
|----|----|
|21.283752575509677|0.7390333340144499|

The following pictures shows the trend of loss Function, Mean Absolute Error, R^2 and actual data compared to predicted data: 
<p align="center">
  <img src="https://github.com/umedsondoniyor/PredictiveMaintenance/blob/master/Output/colab/engine4/model_loss.png"/>
</p>
<p align="center">
  <img src="https://github.com/umedsondoniyor/PredictiveMaintenance/blob/master/Output/colab/engine4/model_MAE.png"/>
</p>
<p align="center">
  <img src="https://github.com/umedsondoniyor/PredictiveMaintenance/blob/master/Output/colab/engine4/model_r2.png"/>
</p>
<p align="center">
  <img src="https://github.com/umedsondoniyor/PredictiveMaintenance/blob/master/Output/colab/engine4/prediction.png"/>
</p>


## References

- Deep Learning for Predictive Maintenance https://github.com/Azure/lstms_for_predictive_maintenance/blob/master/Deep%20Learning%20Basics%20for%20Predictive%20Maintenance.ipynb
- Predictive Maintenance: Step 2A of 3, train and evaluate regression models https://gallery.cortanaintelligence.com/Experiment/Predictive-Maintenance-Step-2A-of-3-train-and-evaluate-regression-models-2
- A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set", NASA Ames Prognostics Data Repository (https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan), NASA Ames Research Center, Moffett Field, CA 
- Understanding LSTM Networks http://colah.github.io/posts/2015-08-Understanding-LSTMs/

