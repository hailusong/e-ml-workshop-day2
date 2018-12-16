# CloudML Lab 1
## The Plan
### Notebook, BigQuery, Cloud DataFlow, Cloud Storage, ML Engine, App
![image][7]
[7]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/S2.png?raw=true


# End to End Baby Weight Lab
## What you learn 

In this lab, you will:

* Explore a large dataset using Datalab and BigQuery

* Export data for machine learning using Cloud Dataflow

* Develop a machine learning model in Tensorflow

* Train a machine learning model at scale on Cloud ML Engine

* Deploy the trained ML model as a microservice

* Invoke the trained model from an App


## Overview 

* In this lab, we walk through the process of building a complete machine learning pipeline covering ingest, exploration, training, evaluation, deployment, and prediction. 

* Along the way, we will discuss how to explore and split large data sets correctly using BigQuery and Cloud Datalab. 

* The machine learning model in TensorFlow will be developed on a small sample locally. 

* The preprocessing operations will be implemented in Cloud Dataflow, so that the same preprocessing can be applied in streaming mode as well. 

* The training of the model will then be distributed and scaled out on Cloud ML Engine. 

* The trained model will be deployed as a microservice and Predictions invoked from a web application.

*This lab consists of 7 parts and will take you about 3 hours*


# Lab 1 
## Exploring a BigQuery dataset to find features and to use in model. 
![image][9] ![image][10]
[9]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/datalab.png?raw=true
[10]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/bq.png?raw=true
*BigQuery: publicdata.samples.natality*


# Lab 2 
## Creating a sampled dataset
![image][10]
[10]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/bq.png?raw=true

*sample the full BQ, create a smaller dataset so that you can use it for model development and local training*


# Lab 3
## Preprocessing using Dataflow
![image][11]
[11]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/df.png?raw=true
*In this lab, you use the high-level Estimator API for a wide-and-deep model*


# Lab 4 
## Preprocessing using Dataflow
![image][11] ![image][12]
[11]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/df.png?raw=true
[12]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/stg.png?raw=true

* While Pandas is great for experimenting, for operationalization of your workflow, it is better to do preprocessing with Apache Beam (Dataflow)


# Lab 5
## Train on Cloud ML Engine
![image][13]
[13]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/mlengine.png?raw=true
*This lab illustrates distributed training and hyperparameter tuning on Cloud ML Engine*


# LUNCH!


# Lab 6
## Deploy and Predict
![image][13]
[13]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/mlengine.png?raw=true
*In this lab, you deploy your trained model to Cloud ML Engine*


# Lab 7
*Interface with your model via an application*


# Break!