
# Getting Started with scikit-learn and XGBoost online predictions  |  Cloud ML Engine for scikit-learn & XGBoost  |  Google Cloud

The Cloud Machine Learning Engine online prediction service manages computing resources in the cloud to run your models. These models can be scikit-learn or XGBoost models that you have trained elsewhere (locally, or via another service) and exported to a file. This page describes the process to get online predictions from these exported models using Cloud ML Engine.

## Overview

In this tutorial, you train a simple model to predict the species of flowers, using the [Iris dataset][1]. After you train and save the model locally, you deploy it to Cloud ML Engine and query it to get online predictions.

This tutorial requires **Python 2.7**. To use Python 3.5, see [how to get online predictions with XGBoost][2] or [how to get online predictions with scikit-learn][3].

## Before you begin

Complete the following steps to set up a GCP account, activate the Cloud ML Engine API, and install and activate the Cloud SDK.


### Set up your environment

Cloud Shell.


### Cloud Shell

1. Open the Google Cloud Platform Console.

[Google Cloud Platform Console][10]

2. Click the **Activate Google Cloud Shell** button at the top of the console window.

![Activate Google Cloud Shell][18]

A Cloud Shell session opens inside a new frame at the bottom of the console and displays a command-line prompt. It can take a few seconds for the shell session to be initialized.

![Cloud Shell session][19]

Your Cloud Shell session is ready to use.

3. Configure the `gcloud` command-line tool to use your selected project. 
    
        gcloud config set project [selected-project-id]

where [`selected-project-id]` is your project ID. (Omit the enclosing brackets.)

### Verify the Google Cloud SDK components

To verify that the Google Cloud SDK components are installed:

1. List your models:
    
        gcloud ml-engine models list

2. If you have not created any models before, the command returns an empty list:
    
        Listed 0 items.

After you start creating models, you can see them listed by using this command.

3. If you have installed `gcloud` previously, update `gcloud`:
    
        gcloud components update

## Install frameworks

### Versions of scikit-learn and XGBoost

Cloud ML Engine runtime versions are updated periodically to include support for new releases of scikit-learn and XGBoost. See the [full details for each runtime version][20].

## Train and save a model

Start by training a simple model for the [Iris dataset][1].

### scikit-learn

Following the [scikit-learn example on model persistence][21], you can train and export a model as shown below:
    
    
    from sklearn import datasets
    from sklearn import svm
    from sklearn.externals import joblib
    
    # Load the Iris dataset
    iris = datasets.load_iris()
    
    # Train a classifier
    classifier = svm.SVC()
    classifier.fit(iris.data, iris.target)
    
    # Export the classifier to a file
    joblib.dump(classifier, 'model.joblib')
    

To export the model, you also have the option to use the [pickle library][22] as follows:
    
    
    import pickle
    with open('model.pkl', 'wb') as model_file:
      pickle.dump(classifier, model_file)
    

### XGBoost

You can export the model by [using the "save_model" method of the Booster object][23].

For the purposes of this tutorial, scikit-learn is used with XGBoost only to import the Iris dataset.
    
    
    from sklearn import datasets
    import xgboost as xgb
    
    # Load the Iris dataset
    iris = datasets.load_iris()
    
    # Load data into DMatrix object
    dtrain = xgb.DMatrix(iris.data, label=iris.target)
    
    # Train XGBoost model
    bst = xgb.train({}, dtrain, 20)
    
    # Export the classifier to a file
    bst.save_model('./model.bst')
    

To export the model, you also have the option to use the [pickle library][22] as follows:
    
    
    import pickle
    with open('model.pkl', 'wb') as model_file:
      pickle.dump(bst, model_file)
    

### Model file naming requirements

The saved model file that you upload to Cloud Storage must be named one of: `model.pkl`, `model.joblib`, or `model.bst`, depending on which library you used. This restriction ensures that Cloud ML Engine uses the same pattern to reconstruct the model on import as was used during export.

### scikit-learn

| Library used to export model | Correct model name |  
| ---------------------------- | ------------------ |  
| `pickle`                     | `model.pkl`        |  
| `joblib`                     | `model.joblib`     |  

### XGBoost

| Library used to export model | Correct model name |  
| ---------------------------- | ------------------ |  
| `pickle`                     | `model.pkl`        |  
| `joblib`                     | `model.joblib`     |  
| `xgboost.Booster`            | `model.bst`        |  

For future iterations of your model, organize your Cloud Storage bucket so that each new model has a dedicated directory.

## Store your model in Cloud Storage

For the purposes of this tutorial, it is easiest to use a dedicated Cloud Storage bucket in the same project you're using for Cloud ML Engine.

If you're using a bucket in a different project, you must ensure that your Cloud ML Engine service account can access your model in Cloud Storage. Without the appropriate permissions, your request to create a Cloud ML Engine model version fails. See more about [granting permissions for storage][24].

### Set up your Cloud Storage bucket

This section shows you how to create a new bucket. You can use an existing bucket, but if it is not part of the project you are using to run Cloud ML Engine, you must explicitly [grant access to the Cloud ML Engine service accounts][25].

1. Specify a name for your new bucket. The name must be unique across all buckets in Cloud Storage.
    
        BUCKET_NAME="your_bucket_name"

For example, use your project name with `-mlengine` appended: 
    
        PROJECT_ID=$(gcloud config list project --format "value(core.project)")
        BUCKET_NAME=${PROJECT_ID}-mlengine

2. Check the bucket name that you created.
    
        echo $BUCKET_NAME

3. Select a region for your bucket and set a `REGION` environment variable.

Warning: You must specify a region (like `us-central1`) for your bucket, not a multi-region location (like `us`). See the [available regions][26] for Cloud ML Engine services. For example, the following code creates `REGION` and sets it to `us-central1`. 
    
        REGION=us-central1

4. Create the new bucket:
    
        gsutil mb -l $REGION gs://$BUCKET_NAME

Note: Use the same region where you plan on running Cloud ML Engine jobs. The example uses `us-central1` because that is the region used in the getting-started instructions.

### Upload the exported model file to Cloud Storage

Run the following command to upload the model you exported earlier in this tutorial, to your bucket in Cloud Storage:
    
    
    gsutil cp ./model.joblib gs://$BUCKET_NAME/model.joblib
    

You can use the same Cloud Storage bucket for multiple model files. Each model file must be within its own directory inside the bucket.

## Format data for prediction

### gcloud

Create an `input.json` file with each input instance on a separate line:
    
    
    [6.8,  2.8,  4.8,  1.4]
    [6.0,  3.4,  4.5,  1.6]
    

Note that the format of input instances needs to match what your model expects. In this example, the Iris model requires 4 features, so your input must be a matrix of shape (`num_instances, 4`).


## Test your model with local predictions

You can use `gcloud` to deploy your model for local predictions. This optional step helps you to save time by sanity-checking your model before deploying it to Cloud ML Engine. Using the model file you uploaded to Cloud Storage, you can run online prediction locally and get a preview of the results the Cloud ML Engine prediction server would return.

Use local prediction with a small subset of your test data to debug a mismatch between training and serving features. For example, if the data you send with your prediction request does not match what your model expects, you can find that out before you incur costs for cloud online prediction requests.

See more about using [`gcloud local predict`][28].

1. Set environment variables for the Cloud Storage directory that contains your model ("gs://your-bucket/"), framework, and the name of your input file, if you have not already done so:
    
        MODEL_DIR="gs://your-bucket/"
        INPUT_FILE="input.json"
        FRAMEWORK="SCIKIT_LEARN"
    

2. Send the prediction request:
    
        gcloud ml-engine local predict --model-dir=$MODEL_DIR 
        --json-instances $INPUT_FILE 
        --framework $FRAMEWORK
    

## Deploy models and versions

Cloud ML Engine organizes your trained models using _model_ and _version_ resources. A Cloud ML Engine model is a container for the versions of your machine learning model.

To deploy a model, you create a model resource in Cloud ML Engine, create a version of that model, then link the model version to the model file stored in Cloud Storage.

### Create a model resource

Cloud ML Engine uses model resources to organize different versions of your model.

### console

1. Open the Cloud ML Engine models page in the GCP Console:

[Open models in the GCP Console][29]

2. If needed, create the model to add your new version to:

    1. Click the **New Model** button at the top of the **Models** page. This brings you to the **Create model** page.

    2. Enter a unique name for your model in the **Model name** box. Optionally, enter a description for your model in the **Description** field.

    3. Click **Save**.

    4. Verify that you have returned to the **Models** page, and that your new model appears in the list.

### gcloud

Create a model resource for your model versions, filling in your desired name for your model without the enclosing brackets:
    
    
        gcloud ml-engine models create "[YOUR-MODEL-NAME]"
    

### REST API

1. Format your request by placing the [model object][30] in the request body. At minimum, you must specify a name for your model. Fill in your desired name for your model without the enclosing brackets:
    
          {"name": "[YOUR-MODEL-NAME]" }
    

2. Make your REST API call to the following path, replacing [`VALUES_IN_BRACKETS]` with the appropriate values:
    
          POST https://ml.googleapis.com/v1/projects/[YOUR-PROJECT-ID]/models/
    

For example, you can make the following request using `cURL`:
    
          curl -X POST -H "Content-Type: application/json" 
        -d '{"name": "[YOUR-MODEL-NAME]"}' 
        -H "Authorization: Bearer `gcloud auth print-access-token`" 
        "https://ml.googleapis.com/v1/projects/[YOUR-PROJECT-ID]/models"
    

You should see output similar to this:
    
          {
        "name": "projects/[YOUR-PROJECT-ID]/models/[YOUR-MODEL-NAME]",
        "regions": [
          "us-central1"
        ]
      }
    

See the [Cloud ML Engine model API][30] for more details.

### Create a model version

Now you are ready to create a model version with the trained model you previously uploaded to Cloud Storage. When you create a version, you need to specify the following parameters:

* `name`: must be unique within the Cloud ML Engine model.
* `deploymentUri`: the path to the directory where the exported model file is stored. Make sure to specify the path to the directory containing the file, not the path to the model file itself. 
    * Path to model \- gs://your_bucket_name/model.pkl
    * Path to directory containing model \- gs://your_bucket_name/
* `framework`: "`SCIKIT_LEARN`" or "`XGBOOST`"
* `runtimeVersion`: must be set to "1.4" or above to ensure you are using a version of Cloud ML Engine that supports scikit-learn and XGBoost.
* `pythonVersion`: must be set to "3.5" to be compatible with model files exported using Python 3. If not set, this defaults to "2.7".

See more information about each of these parameters in [the Cloud ML Engine API for a version resource][31].

See the [full details for each runtime version][20].

### console

1. On the **Models** page, select the name of the model resource you would like to use to create your version. This brings you to the **Model Details** page.

[Open models in the GCP Console][29]

2. Click the **New Version** button at the top of the **Model Details** page. This brings you to the **Create version** page.

3. Enter your version name in the **Name** field. Optionally, enter a description for your version in the **Description** field.

4. Enter the following information about how you trained your model in the corresponding dropdown boxes:

5. Optionally, select a machine type to run online prediction. This field defaults to "Single core CPU".

6. In the **Model URI** field, enter the Cloud Storage bucket location where you uploaded your model file. You may use the "Browse" button to find the correct path. Make sure to specify the path to the directory containing the file, not the path to the model file itself. For example, use "gs://your_bucket_name/" instead of "gs://your_bucket_name/model.pkl".

7. Select a scaling option for online prediction deployment: auto scaling or manual scaling.

    * If you select "Auto scaling", the optional **Minimum number of nodes** field displays. You can enter the minimum number of nodes to keep running at all times, when the service has scaled down. This field defaults to 0.

    * If you select "Manual scaling", the mandatory **Number of nodes** field displays. You must enter the number of nodes you want to keep running at all times.

Learn more about [pricing for prediction costs][32].

8. To finish creating your model version, click **Save**.

### gcloud

1. Set environment variables to store the path to the Cloud Storage directory where your model binary is located, your model name, your version name and your framework choice ("SCIKIT_LEARN" or "XGBOOST"). Replace [`VALUES_IN_BRACKETS]` with the appropriate values:
    
          MODEL_DIR="gs://your_bucket_name/"
      VERSION_NAME="[YOUR-VERSION-NAME]"
      MODEL_NAME="[YOUR-MODEL-NAME]"
      FRAMEWORK="SCIKIT_LEARN"
    

2. Create the version:
    
          gcloud ml-engine versions create $VERSION_NAME 
          --model $MODEL_NAME --origin $MODEL_DIR 
          --runtime-version=1.10 --framework $FRAMEWORK 
          --python-version=3.5
    

Creating the version takes a few minutes. When it is ready, you should see the following output:
    
          Creating version (this might take a few minutes)......done.
    

3. Get information about your new version:
    
          gcloud ml-engine versions describe $VERSION_NAME 
          --model $MODEL_NAME
    

You should see output similar to this:
    
          createTime: '2018-02-28T16:30:45Z'
      deploymentUri: gs://your_bucket_name
      framework: SCIKIT_LEARN
      machineType: mls1-highmem-1
      name: projects/[YOUR-PROJECT-ID]/models/[YOUR-MODEL-NAME]/versions/[YOUR-VERSION-NAME]
      pythonVersion: '3.5'
      runtimeVersion: '1.10'
      state: READY
    

### REST API

1. Format your request body to contain the [version object][31]. This example specifies the version `name`, `deploymentUri`, `runtimeVersion` and `framework`. Replace [`VALUES_IN_BRACKETS]` with the appropriate values:
    
          {
        "name": "[YOUR-VERSION-NAME]",
        "deploymentUri": "gs://your_bucket_name/"
        "runtimeVersion": "1.10"
        "framework": "SCIKIT_LEARN"
        "pythonVersion": "3.5"
      }
    

2. Make your REST API call to the following path, replacing [`VALUES_IN_BRACKETS]` with the appropriate values:
    
          POST https://ml.googleapis.com/v1/projects/[YOUR-PROJECT-ID]/models/[YOUR-MODEL-NAME]/versions
    

For example, you can make the following request using `cURL`:
    
            curl -X POST -H "Content-Type: application/json" 
          -d '{"name": "[YOUR-VERSION-NAME]", "deploymentUri": "gs://your_bucket_name/", "runtimeVersion": "1.10", "framework": "SCIKIT_LEARN", "pythonVersion": "3.5"}' 
          -H "Authorization: Bearer `gcloud auth print-access-token`" 
          "https://ml.googleapis.com/v1/projects/[YOUR-PROJECT-ID]/models/[YOUR-MODEL-NAME]/versions"
    

Creating the version takes a few minutes. When it is ready, you should see output similar to this:
    
          {
        "name": "projects/[YOUR-PROJECT-ID]/operations/create[_YOUR-MODEL-NAME][_YOUR-VERSION-NAME]-[TIMESTAMP]",
        "metadata": {
          "@type": "type.googleapis.com/google.cloud.ml.v1.OperationMetadata",
          "createTime": "2018-07-07T02:51:50Z",
          "operationType": "CREATE_VERSION",
          "modelName": "projects/[YOUR-PROJECT-ID]/models/[YOUR-MODEL-NAME]",
          "version": {
            "name": "projects/[YOUR-PROJECT-ID]/models/[YOUR-MODEL-NAME]/versions/[YOUR-VERSION-NAME]",
            "deploymentUri": "gs://your_bucket_name",
            "createTime": "2018-07-07T02:51:49Z",
            "runtimeVersion": "1.10",
            "framework": "SCIKIT_LEARN",
            "machineType": "mls1-highmem-1",
            "pythonVersion": "3.5"
          }
        }
      }
    

## Send online prediction request

After you have successfully created a model version, Cloud ML Engine starts a new server that is ready to serve prediction requests.

### gcloud

1. Set environment variables for your model name, version name, and the name of your input file:
    
        MODEL_NAME="iris"
    VERSION_NAME="v1"
    INPUT_FILE="input.json"
    

2. Send the prediction request:
    
        gcloud ml-engine predict --model $MODEL_NAME --version 
      $VERSION_NAME --json-instances $INPUT_FILE
    

### Python

This sample assumes that you are familiar with the Google Cloud Client library for Python. If you aren't familiar with it, see [Using the Python Client Library][33].
    
    
    import googleapiclient.discovery
    
    def predict_json(project, model, instances, version=None):
        """Send json data to a deployed model for prediction.
        Args:
            project (str): project where the Cloud ML Engine Model is deployed.
            model (str): model name.
            instances ([[float]]): List of input instances, where each input
               instance is a list of floats.
            version: str, version of the model to target.
        Returns:
            Mapping[str: any]: dictionary of prediction results defined by the
                model.
        """
        # Create the ML Engine service object.
        # To authenticate set the environment variable
        # GOOGLE_APPLICATION_CREDENTIALS=
        service = googleapiclient.discovery.build('ml', 'v1')
        name = 'projects/{}/models/{}'.format(project, model)
    
        if version is not None:
            name += '/versions/{}'.format(version)
    
        response = service.projects().predict(
            name=name,
            body={'instances': instances}
        ).execute()
    
        if 'error' in response:
            raise RuntimeError(response['error'])
    
        return response['predictions']
    

See more information about each of these parameters in [the Cloud ML Engine API for prediction input][34].

## What's next

[1]: https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
[2]: https://cloud.google.com/ml-engine/docs/scikit/getting-predictions-xgboost
[3]: https://cloud.google.com/ml-engine/docs/scikit/using-pipelines-for-preprocessing
[4]: https://accounts.google.com/Login
[5]: https://accounts.google.com/SignUp
[6]: https://console.cloud.google.com/cloud-resource-manager
[7]: https://cloud.google.com/billing/docs/how-to/modify-project
[8]: https://console.cloud.google.com/flows/enableapi?apiid=ml.googleapis.com,compute_component
[9]: https://console.cloud.google.com/apis/credentials/serviceaccountkey
[10]: https://console.cloud.google.com/
[11]: https://cloud.google.com/iam/docs/granting-roles-to-service-accounts
[12]: https://cloud.google.com/sdk/docs/
[13]: https://www.python.org/downloads/
[14]: https://pip.pypa.io/en/stable/installing/#installing-with-get-pip-py
[15]: https://pip.pypa.io/en/stable/installing/#upgrading-pip
[16]: https://virtualenv.pypa.io/en/stable/installation/
[17]: https://virtualenv.pypa.io/en/stable/userguide/
[18]: https://cloud.google.com/shell/docs/images/shell_icon.png
[19]: https://cloud.google.com/shell/docs/images/new-console.png
[20]: https://cloud.google.com/ml-engine/docs/scikit/runtime-version-list
[21]: https://scikit-learn.org/stable/modules/model_persistence.html
[22]: https://docs.python.org/3/library/pickle.html
[23]: http://xgboost.readthedocs.io/en/latest/python/python_intro.html#training
[24]: https://cloud.google.com/ml-engine/docs/scikit/working-with-cloud-storage
[25]: https://cloud.google.com/ml-engine/docs/working-with-cloud-storage#setup-different-project
[26]: https://cloud.google.com/ml-engine/docs/regions
[27]: https://cloud.google.com/ml-engine/docs/v1/predict-request#request-body
[28]: https://cloud.google.com/sdk/gcloud/reference/ml-engine/local/predict
[29]: https://console.cloud.google.com/mlengine/models
[30]: https://cloud.google.com/ml-engine/reference/rest/v1/projects.models#Model
[31]: https://cloud.google.com/ml-engine/reference/rest/v1/projects.models.versions
[32]: https://cloud.google.com/ml-engine/docs/scikit/pricing#more_about_prediction_costs
[33]: https://cloud.google.com/ml-engine/docs/python-guide
[34]: https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#predictioninput

  