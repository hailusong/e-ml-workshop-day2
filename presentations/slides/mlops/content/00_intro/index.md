![](../images/arctiq-logo-white.png)<!-- .element style="border: 0; background: None; box-shadow: None height="10%" width="10%"" -->

# Machine Learning on GCP - Day 2


# ML Engine
## GCP ML Engine
### What is CloudML
![image][1]
[1]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/S1.png?raw=true


## Focus on models, not operations.
Google Cloud Machine Learning (ML) Engine is:
* A managed service that enables developers and data scientists to build and bring superior machine learning models to production.
* Cloud ML Engine offers training and prediction services, which can be used together or individually.


## Train
* Machine learning involves training a computer model to find patterns in data. 

* The more high-quality data that you train a well-designed model with, the more intelligent your solution will be. 

* You can build your models with multiple ML frameworks (in beta), including scikit-learn, XGBoost, Keras, and TensorFlow

* Cloud ML Engine enables you to automatically design and evaluate model architectures to achieve an intelligent solution faster and without experts. 

* Cloud ML Engine scales to leverage all your data. It can train any model at large scale on a managed cluster.


## Predict
* Prediction incorporates intelligence into your applications and workflows. Once you have a trained model, prediction applies what the computer learned to new examples. ML Engine offers two types of prediction:

* Online Prediction deploys ML models with serverless, fully managed hosting that responds in real time with high availability. Our global prediction platform automatically scales to adjust to any throughput. It provides a secure web endpoint to integrate ML into your applications.

* Batch Prediction offers cost-effective inference with unparalleled throughput for asynchronous applications. It scales to perform inference on TBs of production data.


## Train and deploy multiple frameworks
* Training and Online Prediction allow developers and data scientists to use multiple ML frameworks, and seamlessly deploy ML models into production 

* No Docker container required. Users can also import models that have been trained anywhere.

![image][2]  ![image][3]  ![image][4]  ![image][5]  
[2]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/tensorflow-logo.png?raw=true
[3]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/keras-logo.png?raw=true
[4]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/scikit-learn-logo.png?raw=true
[5]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/xgboost-logo.png?raw=true


# CloudML Features
## Automatic Resource Provisioning:

* Focus on model development and deployment without worrying about infrastructure. The managed service automates all resource provisioning and monitoring. 

* Build models using managed distributed training infrastructure that supports CPUs, GPUs, and TPUs. 

* Accelerate model development by training across many nodes or running multiple experiments in parallel


## HyperTune 
* Achieve superior results faster by automatically tuning deep learning hyperparameters with HyperTune. Data scientists can manage thousands of tuning experiments on the cloud. This saves many hours of tedious and error-prone work.


## Server-Side Preprocessing 
*  Push deployment preprocessing to Google Cloud with scikit-learn pipelines and tf.transform. This means that you can send raw data to models in production and reduce local computation. This also prevents data skew being introduced through different preprocessing in training and prediction.


## Portable Models 

* Use the open source TensorFlow SDK,  to train models locally on sample data sets and use the Google Cloud Platform for training at scale. Models trained using Cloud ML Engine can be downloaded for local execution or mobile integration. 

* You can also import scikit-learn, XGBoost, Keras, and TensorFlow models that have been trained anywhere for fully-managed, real-time prediction hosting — no Docker container required.


## Integrated 

* Google services are designed to work together. Cloud ML Engine works with Cloud Dataflow for feature processing and Cloud Storage for data storage.


## Refresher
![image][6]
[6]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/refresher.png?raw=true


# Get Connected and Grab a Coffee
