# Machine Learn on GCP - Day 2

# ML Engine

## GCP ML Engine

### What is CloudML
![image][1]

### Focus on models, not operations.
Google Cloud Machine Learning (ML) Engine is:

* A managed service that enables developers and data scientists to build and bring superior machine learning models to production.

* Cloud ML Engine offers training and prediction services, which can be used together or individually.

** Train**

* Machine learning involves training a computer model to find patterns in data. 

* The more high-quality data that you train a well-designed model with, the more intelligent your solution will be. 

* You can build your models with multiple ML frameworks (in beta), including scikit-learn, XGBoost, Keras, and TensorFlow

* Cloud ML Engine enables you to automatically design and evaluate model architectures to achieve an intelligent solution faster and without experts. 

* Cloud ML Engine scales to leverage all your data. It can train any model at large scale on a managed cluster.

** Predict**

* Prediction incorporates intelligence into your applications and workflows. Once you have a trained model, prediction applies what the computer learned to new examples. ML Engine offers two types of prediction:

* Online Prediction deploys ML models with serverless, fully managed hosting that responds in real time with high availability. Our global prediction platform automatically scales to adjust to any throughput. It provides a secure web endpoint to integrate ML into your applications.

* Batch Prediction offers cost-effective inference with unparalleled throughput for asynchronous applications. It scales to perform inference on TBs of production data.

** Train and deploy multiple frameworks.** 

* Training and Online Prediction allow developers and data scientists to use multiple ML frameworks, and seamlessly deploy ML models into production 

* No Docker container required. Users can also import models that have been trained anywhere.

![image][2]  ![image][3]  ![image][4]  ![image][5]  


### CloudML Features
** Automatic Resource Provisioning: **

* Focus on model development and deployment without worrying about infrastructure. The managed service automates all resource provisioning and monitoring. 
* Build models using managed distributed training infrastructure that supports CPUs, GPUs, and TPUs. 
* Accelerate model development by training across many nodes or running multiple experiments in parallel

** HyperTune **

* Achieve superior results faster by automatically tuning deep learning hyperparameters with HyperTune. Data scientists can manage thousands of tuning experiments on the cloud. This saves many hours of tedious and error-prone work.

** Server-Side Preprocessing **

*  Push deployment preprocessing to Google Cloud with scikit-learn pipelines and tf.transform. This means that you can send raw data to models in production and reduce local computation. This also prevents data skew being introduced through different preprocessing in training and prediction.
### Local implementation


** Portable Models **

* Use the open source TensorFlow SDK,  to train models locally on sample data sets and use the Google Cloud Platform for training at scale. Models trained using Cloud ML Engine can be downloaded for local execution or mobile integration. 

* You can also import scikit-learn, XGBoost, Keras, and TensorFlow models that have been trained anywhere for fully-managed, real-time prediction hosting — no Docker container required.

** Integrated **

* Google services are designed to work together. Cloud ML Engine works with Cloud Dataflow for feature processing and Cloud Storage for data storage.

## Refresher
![image][6]


# Get Connected  & Break!

# CloudML Lab 1

## The Plan
### Notebook, BigQuery, Cloud DataFlow, Cloud Storage, ML Engine, App
![image][7]

### End to End Baby Weight Lab

** What you learn **

In this lab, you will:

* Explore a large dataset using Datalab and BigQuery
* Export data for machine learning using Cloud Dataflow
* Develop a machine learning model in Tensorflow
* Train a machine learning model at scale on Cloud ML Engine
* Deploy the trained ML model as a microservice
* Invoke the trained model from an App

### Overview 

* In this lab, we walk through the process of building a complete machine learning pipeline covering ingest, exploration, training, evaluation, deployment, and prediction. 

* Along the way, we will discuss how to explore and split large data sets correctly using BigQuery and Cloud Datalab. 

* The machine learning model in TensorFlow will be developed on a small sample locally. 

* The preprocessing operations will be implemented in Cloud Dataflow, so that the same preprocessing can be applied in streaming mode as well. 

* The training of the model will then be distributed and scaled out on Cloud ML Engine. 

* The trained model will be deployed as a microservice and 
Predictions invoked from a web application.

** This lab consists of 7 parts and will take you about 3 hours. **

# Lab 1 
## Exploring a BigQuery dataset to find features and to use in model. 
![image][9] ![image][10]

** BigQuery: publicdata.samples.natality **

# Lab 2 
## Creating a sampled dataset
![image][10]

 ** sample the full BQ, create a smaller dataset so that you can use it for model development and local training **

# Lab 3
## Preprocessing using Dataflow
![image][11]

** In this lab, you use the high-level Estimator API for a wide-and-deep model **

# Deeper into Neural Nets 

### CNN’s: What Are they Used For?
* Images that you want to find features in
* Machine Translations
* Sentiment Analysis

### They can find features that aren’t in a specific spot
* Like a stop sign in a picture
* Or words within a sentence

## Convolutional Neural Networks

* Convolutional neural networks are deep artificial neural networks that are used primarily to classify images (e.g. name what they see), cluster them by similarity (photo search), and perform object recognition within scenes. They are algorithms that can identify faces, individuals, street signs, tumors, platypuses and many other aspects of visual data.
![image][8]


# CNN DEMO

# Lab 4 
## Preprocessing using Dataflow

![image][11] ![image][12]

** While Pandas is great for experimenting, for operationalization of your workflow, it is better to do preprocessing with Apache Beam (Dataflow) **
 
# Lab 5
## Train on Cloud ML Engine

![image][13]

** This lab illustrates distributed training and hyperparameter tuning on Cloud ML Engine **

# LUNCH!

# Lab 6
## Deploy and Predict

![image][13]

** In this lab, you deploy your trained model to Cloud ML Engine **

# Lab 7
** Interface with your model via an application **

# Break!

# CloudML Lab 2

## Overview

This lab provides an introductory, end-to-end walkthrough of training and prediction on Cloud Machine Learning Engine. You will walk through a sample that uses a census dataset to:

* Create a TensorFlow training application and validate it locally.
* Run your training job on a single worker instance in the cloud.
* Run your training job as a distributed training job in the cloud.
* Optimize your hyperparameters by using hyperparameter tuning.
* Deploy a model to support prediction.
* Request an online prediction and see the response.

# Break!

# GANs
### Generative Adversarial Network (GAN)

* One of the most promising recent developments in Deep Learning

* Attacks the problem of unsupervised learning by training two deep networks, that compete and cooperate with each other. In the course of training, both networks eventually learn how to perform their tasks.

* GAN is almost always explained like the case of a counterfeiter (Generative) and the police (Discriminator)

### Discriminator

![image][14]

A discriminator that tells how real an image is, is basically a deep Convolutional Neural Network (CNN) 

### Generator

![image][15]

The generator synthesizes fake images

### Generative Adversarial Network (GAN)

![image][16]

### Example
The GAN is learning how to write handwritten digits on its own!

![image][17]

### Faces
![image][18]

### Video Demo
[![Faces Demo](http://img.youtube.com/vi/XOxxPcy5Gr4/0.jpg)](http://www.youtube.com/watch?v=XOxxPcy5Gr4)


# Using Existing Models 
## Applications

Keras Applications are deep learning models that are made available alongside pre-trained weights. These models can be used for prediction, feature extraction, and fine-tuning. Weights are downloaded automatically when instantiating a model. 

Available models for image classification with weights trained on ImageNet:

* Xception
* VGG16
* VGG19
* ResNet50
* InceptionV3
* InceptionResNetV2
* MobileNet
* DenseNet
* NASNet
* MobileNetV2

# TF - Object Detection
## Lab 3

# Questions / Discussion

# Done! You Made It! 

[1]: https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/S1.png?raw=true

[2]: https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/tensorflow-logo.png?raw=true

[3]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/keras-logo.png?raw=true

[4]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/scikit-learn-logo.png?raw=true

[5]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/xgboost-logo.png?raw=true
  
[6]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/refresher.png?raw=true

[7]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/S2.png?raw=true

[8]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/cnn.png?raw=true

[9]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/datalab.png?raw=true

[10]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/bq.png?raw=true

[11]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/df.png?raw=true

[12]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/stg.png?raw=true

[13]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/mlengine.png?raw=true

[14]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/gan1.png?raw=true

[15]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/gan2.png?raw=true

[16]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/gan3.png?raw=true

[17]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/gan4.png?raw=true

[18]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/gan5.png?raw=true
