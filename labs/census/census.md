## Overview

This document provides an introductory, end-to-end walkthrough of training and prediction on Cloud Machine Learning Engine. You will walk through a sample that uses a census dataset to:

* Create a TensorFlow training application and validate it locally.
* Run your training job on a single worker instance in the cloud.
* Run your training job as a distributed training job in the cloud.
* Optimize your hyperparameters by using hyperparameter tuning.
* Deploy a model to support prediction.
* Request an online prediction and see the response.
* Request a batch prediction.

## What you will build

The sample builds a wide and deep model for predicting income category based on United States Census Income Dataset. The two income categories (also known as labels) are:

* **>50K**—Greater than 50,000 dollars
* **<=50K**—Less than or equal to 50,000 dollars

Wide and deep models use deep neural nets (DNNs) to learn high-level abstractions about complex features or interactions between such features. These models then combine the outputs from the DNN with a linear regression performed on simpler features. This provides a balance between power and speed that is effective on many structured data problems.

### Cloud Shell

1. Open the Google Cloud Platform Console.

[Google Cloud Platform Console][9]

2. Click the **Activate Google Cloud Shell** button at the top of the console window.

![Activate Google Cloud Shell][17]

A Cloud Shell session opens inside a new frame at the bottom of the console and displays a command-line prompt. It can take a few seconds for the shell session to be initialized.

![Cloud Shell session][18]

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


### Run a simple TensorFlow Python program

Run a simple Python program to test your installation of TensorFlow. If you are using Cloud Shell, note that TensorFlow is already installed.

1. Start a Python interactive shell.
    
        python
    

2. Import TensorFlow.
    
        >>> import tensorflow as tf

3. Create a constant that contains a string.
    
        >>> hello = tf.constant('Hello, TensorFlow!')

4. Create a TensorFlow session.
    
        >>> sess = tf.Session()

You can ignore the warnings that the TensorFlow library wasn't compiled to use certain instructions.

5. Display the value of `hello`.
    
        >>> print(sess.run(hello))

If successful, the system outputs: 
    
        Hello, TensorFlow!

6. Stop the Python interactive shell. 
    
        >>> exit()

## Python version support

Cloud ML Engine runs Python 2.7 by default, and the sample for this tutorial uses Python 2.7.

Python 3.5 is available for training when you use Cloud ML Engine runtime version 1.4 or greater. Online and batch prediction work with trained models, regardless of whether they were trained using Python 2 or Python 3.

See how to [submit a training job using Python 3.5][21].

## Download the code for this tutorial
  
###  Cloud Shell 

1. Enter the following command to download the Cloud ML Engine sample zip file:
    
        wget https://github.com/GoogleCloudPlatform/cloudml-samples/archive/master.zip
    

2. Unzip the file to extract the `cloudml-samples-master` directory.
    
        unzip master.zip
    

3. Navigate to the `cloudml-samples-master > census > estimator` directory. The commands in this walkthrough must be run from the `estimator` directory.
    
        cd cloudml-samples-master/census/estimator
    

## Develop and validate your training application locally

Before you run your training application in the cloud, get it running locally. Local environments provide an efficient development and validation workflow so that you can iterate quickly. You also won't incur charges for cloud resources when debugging your application locally.

### Get your training data

The relevant data files, `adult.data` and `adult.test`, are hosted in a public Cloud Storage bucket. For purposes of this sample, use the versions on Cloud Storage, which have undergone some trivial cleaning, instead of the original source data. See below for more information [about the data][23].

You can read the data files directly from Cloud Storage or copy them to your local environment. For purposes of this sample you will download the samples for local training, and later upload them to your own Cloud Storage bucket for cloud training.

1. Download the data to a local file directory and set variables that point to the downloaded data files.
    
        mkdir data
        gsutil -m cp gs://cloud-samples-data/ml-engine/census/data/* data/
    

2. Set the `TRAIN_DATA` AND `EVAL_DATA` variables to your local file paths. For example, the following commands set the variables to local paths.
    
        TRAIN_DATA=$(pwd)/data/adult.data.csv
        EVAL_DATA=$(pwd)/data/adult.test.csv
    


###  Cloud Shell 

Although TensorFlow is installed on Cloud Shell, you must run the sample's `requirements.txt` file to ensure you are using the same version of TensorFlow required by the sample.
    
    
      pip install --user -r ../requirements.txt
    

Running this command will install TensorFlow 1.10, which is used in the tutorial.

### Run a local training job

A local training job loads your Python training program and starts a training process in an environment that's similar to that of a live Cloud ML Engine cloud training job.

1. Specify an output directory and set a `MODEL_DIR` variable. The following command sets `MODEL_DIR` to a value of `output`.
    
        MODEL_DIR=output
    

2. It's a good practice to delete the contents of the output directory in case data remains from a previous training run. The following command deletes all data in the `output` directory.

**Caution:** Be careful when using `rm -rf`. Check the variable name, and use `rm -i` if you want the system to ask you before deleting each file.
    
        rm -rf $MODEL_DIR/*
    

3. To run your training locally, run the following command:
    
        gcloud ml-engine local train \
        --module-name trainer.task \
        --package-path trainer/ \
        --job-dir $MODEL_DIR \
        -- \
        --train-files $TRAIN_DATA \
        --eval-files $EVAL_DATA \
        --train-steps 1000 \
        --eval-steps 100
    

By default, verbose logging is turned off. You can enable it by setting the `\--verbosity` tag to `DEBUG`. A later example shows you how to enable it.

### Inspect the summary logs using Tensorboard

To see the evaluation results, you can use the visualization tool called [TensorBoard][25]. With TensorBoard, you can visualize your TensorFlow graph, plot quantitative metrics about the execution of your graph, and show additional data like images that pass through the graph. Tensorboard is available as part of the TensorFlow installation.

Follow the steps below to launch TensorBoard and point it at the summary logs produced during training, both during and after execution.


###  Cloud Shell 

1. Launch TensorBoard:
    
        tensorboard --logdir=$MODEL_DIR --port=8080
    

2. Select "Preview on port 8080" from the [Web Preview][26] menu at the top of the command line.

Click on **Accuracy** to see graphical representations of how accuracy changes as your job progresses.

![Tensorboard accuracy graphs][27]

You can shut down TensorBoard at any time by typing `ctrl+c` on the command line.

### Run a local training job in distributed mode

You can test whether your model works with the Cloud ML Engine's distributed execution environment by running a local training job using the `\--distributed` flag.

1. Specify an output directory and set `MODEL_DIR` variable again. The following command sets `MODEL_DIR` to a value of `output-dist`.
    
        MODEL_DIR=output-dist
    

2. Delete the contents of the `output` directory in case data remains from a previous training run.

**Caution:** Be careful when using `rm -rf`. Check the variable name, and use `rm -i` if you want the system to ask you before deleting each file.
    
        rm -rf $MODEL_DIR/*
    

3. Run the `local train` command using the `\--distributed` option. Be sure to place the flag above the `\--` that separates the user arguments from the command-line arguments.
    
        gcloud ml-engine local train \
        --module-name trainer.task \
        --package-path trainer/ \
        --job-dir $MODEL_DIR \
        --distributed \
        -- \
        --train-files $TRAIN_DATA \
        --eval-files $EVAL_DATA \
        --train-steps 1000 \
        --eval-steps 100
    

### Inspect the output

Output files are written to the directory specified by `\--job-dir`, which was set to `output-dist`:
    
    
    ls -R output-dist/
    

You should see output similar to this:
    
    
    checkpoint
    eval
    events.out.tfevents.1488577094.
    export
    graph.pbtxt
    model.ckpt-1000.data-00000-of-00001
    model.ckpt-1000.index
    model.ckpt-1000.meta
    model.ckpt-2.data-00000-of-00001
    model.ckpt-2.index
    model.ckpt-2.meta
    
    output-dist//eval:
    events.out.tfevents..
    events.out.tfevents.
    events.out.tfevents..
    
    output-dist//export:
    census
    
    output-dist//export/census:
    
    
    output-dist//export/census/:
    saved_model.pb
    variables
    ...
    
    

### Inspect the logs

Inspect the summary logs using Tensorboard the same way that you did for the single-instance training job except that you must change the `\--logdir` value to match the output directory name you used for distributed mode.


###  Cloud Shell 

1. Launch TensorBoard:
    
        tensorboard --logdir=$MODEL_DIR --port=8080
    

2. Select "Preview on port 8080" from the [Web Preview][26] menu at the top of the command line.

## Set up your Cloud Storage bucket

This section shows you how to create a new bucket. You can use an existing bucket, but if it is not part of the project you are using to run Cloud ML Engine, you must explicitly [grant access to the Cloud ML Engine service accounts][28].

1. Specify a name for your new bucket. The name must be unique across all buckets in Cloud Storage.
    
        BUCKET_NAME="your_bucket_name"

For example, use your project name with `-mlengine` appended: 
    
        PROJECT_ID=$(gcloud config list project --format "value(core.project)")
        BUCKET_NAME=${PROJECT_ID}-mlengine

2. Check the bucket name that you created.
    
        echo $BUCKET_NAME

3. Select a region for your bucket and set a `REGION` environment variable.

Warning: You must specify a region (like `us-central1`) for your bucket, not a multi-region location (like `us`). See the [available regions][29] for Cloud ML Engine services. For example, the following code creates `REGION` and sets it to `us-central1`. 
    
        REGION=us-central1

4. Create the new bucket:
    
        gsutil mb -l $REGION gs://$BUCKET_NAME

Note: Use the same region where you plan on running Cloud ML Engine jobs. The example uses `us-central1` because that is the region used in the getting-started instructions.

Upload the data files to your Cloud Storage bucket.

1. Use `gsutil` to copy the two files to your Cloud Storage bucket.
    
        gsutil cp -r data gs://$BUCKET_NAME/data
    

2. Set the `TRAIN_DATA` and `EVAL_DATA` variables to point to the files.
    
        TRAIN_DATA=gs://$BUCKET_NAME/data/adult.data.csv
        EVAL_DATA=gs://$BUCKET_NAME/data/adult.test.csv
    

3. Use `gsutil` again to copy the JSON test file `test.json` to your Cloud Storage bucket.
    
        gsutil cp ../test.json gs://$BUCKET_NAME/data/test.json
    

4. Set the `TEST_JSON` variable to point to that file.
    
        TEST_JSON=gs://$BUCKET_NAME/data/test.json
    

## Run a single-instance training job in the cloud

With a validated training job that runs in both single-instance and distributed mode, you're now ready to run a training job in the cloud. You'll start by requesting a single-instance training job.

Use the default `BASIC` [scale tier][30] to run a single-instance training job. The initial job request can take a few minutes to start, but subsequent jobs run more quickly. This enables quick iteration as you develop and validate your training job.

1. Select a name for the initial training run that distinguishes it from any subsequent training runs. For example, you can append a number to represent the iteration.
    
        JOB_NAME=census_single_1
    

2. Specify a directory for output generated by Cloud ML Engine by setting an `OUTPUT_PATH` variable to include when requesting training and prediction jobs. The `OUTPUT_PATH` represents the fully qualified Cloud Storage location for model checkpoints, summaries, and exports. You can use the `BUCKET_NAME` variable you defined in a previous step.

It's a good practice to use the job name as the output directory. For example, the following `OUTPUT_PATH` points to a directory named `census_single_1`.
    
        OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
    

3. Run the following command to submit a training job in the cloud that uses a single process. This time, set the `\--verbosity` tag to `DEBUG` so that you can inspect the full logging output and retrieve accuracy, loss, and other metrics. The output also contains a number of other warning messages that you can ignore for the purposes of this sample.
    
        gcloud ml-engine jobs submit training $JOB_NAME \
        --job-dir $OUTPUT_PATH \
        --runtime-version 1.10 \
        --module-name trainer.task \
        --package-path trainer/ \
        --region $REGION \
        -- \
        --train-files $TRAIN_DATA \
        --eval-files $EVAL_DATA \
        --train-steps 1000 \
        --eval-steps 100 \
        --verbosity DEBUG
    

You can monitor the progress of your training job by watching the command-line output or in **ML Engine** > **Jobs** on [Google Cloud Platform Console][31].

### Inspect the output

In cloud training, outputs are produced into Cloud Storage. In this sample, outputs are saved to `OUTPUT_PATH`; to list them, run:
    
    
    gsutil ls -r $OUTPUT_PATH
    

The outputs should be similar to the [outputs from training locally (above)][32].

### Inspect the Stackdriver logs

Logs are a useful way to understand the behavior of your training code on the cloud. When Cloud ML Engine runs a training job, it captures all `stdout` and `stderr` streams and logging statements. These logs are stored in Stackdriver Logging; they are visible both during and after execution.

The easiest way to find the logs for your job is to select your job in **ML Engine** > **Jobs** on [GCP Console][31], and then click "View logs".

If you leave "All logs" selected, you see all logs from all workers. You can also select a specific task; `master-replica-0` gives you an overview of the job's execution from the master's perspective.

Because you selected verbose logging, you can inspect the full logging output. Look for the term `accuracy` in the logs:

![screenshot of Stackdriver logging console for ML Engine jobs][33]

If you want to view these logs in your terminal, you can do so from the command line with:
    
    
    gcloud ml-engine jobs stream-logs $JOB_NAME
    

See all the options for [gcloud ml-engine jobs stream-logs][34].

### Inspect the summary logs using Tensorboard

You can inspect the behavior of your training job by launching [TensorBoard][25] and pointing it at the summary logs produced during training — both during and after execution.

Because the training programs write summaries directly to a Cloud Storage location, Tensorboard can automatically read from them without manual copying of event files.


###  Cloud Shell 

1. Launch TensorBoard:
    
        tensorboard --logdir=$OUTPUT_PATH --port=8080
    

2. Select "Preview on port 8080" from the [Web Preview][26] menu at the top of the command line.

Click on **Accuracy** to see graphical representations of how accuracy changes as your job progresses.

You can shut down TensorBoard at any time by typing `ctrl+c` on the command line.

## Run distributed training in the cloud

To take advantage of Google's scalable infrastructure when running training jobs, configure your training job to run in distributed mode.

No code changes are necessary to run this model as a distributed process in Cloud ML Engine.

To run a distributed job, set [`\--scale-tier`][35] to any tier above basic. For more information about scale tiers, see the [scale tier documentation][30].

1. Select a name for your distributed training job that distinguishes it from other training jobs. For example, you could use `dist` to represent distributed and a number to represent the iteration.
    
        JOB_NAME=census_dist_1
    

2. Specify `OUTPUT_PATH` to include the job name so that you don't inadvertently reuse checkpoints between jobs. You might have to redefine `BUCKET_NAME` if you've started a new command-line session since you last defined it. For example, the following `OUTPUT_PATH` points to a directory named `census-dist-1`.
    
        OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
    

3. Run the following command to submit a training job in the cloud that uses multiple workers. Note that the job can take a few minutes to start.

Place [`\--scale-tier`][35] above the `\--` that separates the user arguments from the command-line arguments. For example, the following command uses a scale tier of `STANDARD_1`:
    
        gcloud ml-engine jobs submit training $JOB_NAME \
        --job-dir $OUTPUT_PATH \
        --runtime-version 1.10 \
        --module-name trainer.task \
        --package-path trainer/ \
        --region $REGION \
        --scale-tier STANDARD_1 \
        -- \
        --train-files $TRAIN_DATA \
        --eval-files $EVAL_DATA \
        --train-steps 1000 \
        --verbosity DEBUG  \
        --eval-steps 100
    

You can monitor the progress of your job by watching the command-line output or in **ML Engine** > **Jobs** on [Google Cloud Platform Console][31].

### Inspect the logs

Inspect the Stackdriver logs and summary logs the same way that you did for the single-instance training job.

For Stackdriver logs: Either select your job in **ML Engine** > **Jobs** on [GCP Console][31], and then click **View logs** or use the following command from your terminal:
    
    
    gcloud ml-engine jobs stream-logs $JOB_NAME
    


### Hyperparameter Tuning

Cloud ML Engine offers hyperparameter tuning to help you maximize your model's predictive accuracy. The census sample stores the hyperparameter configuration settings in a YAML file named `hptuning_config.yaml` and includes the file in the training request using the `\--config` variable.

1. Select a new job name and create a variable that references the configuration file.
    
        HPTUNING_CONFIG=../hptuning_config.yaml
        JOB_NAME=census_core_hptune_1
        TRAIN_DATA=gs://$BUCKET_NAME/data/adult.data.csv
        EVAL_DATA=gs://$BUCKET_NAME/data/adult.test.csv
    

2. Specify `OUTPUT_PATH` to include the job name so that you don't inadvertently reuse checkpoints between jobs. You might have to redefine `BUCKET_NAME` if you've started a new command-line session since you last defined it. For example, the following `OUTPUT_PATH` points to a directory named `census_core_hptune_1`.
    
        OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
    

3. Run the following command to submit a training job that not only uses multiple workers but also uses hyperparameter tuning.
    
        gcloud ml-engine jobs submit training $JOB_NAME \
        --stream-logs \
        --job-dir $OUTPUT_PATH \
        --runtime-version 1.10 \
        --config $HPTUNING_CONFIG \
        --module-name trainer.task \
        --package-path trainer/ \
        --region $REGION \
        --scale-tier STANDARD_1 \
        -- \
        --train-files $TRAIN_DATA \
        --eval-files $EVAL_DATA \
        --train-steps 1000 \
        --verbosity DEBUG  \
        --eval-steps 100
    

For more information about hyperparameter tuning, see the [hyperparameter tuning overview][36].

## Deploy a model to support prediction

1. Choose a name for your model; this must start with a letter and contain only letters, numbers, and underscores. For example:
    
        MODEL_NAME=census
    

2. Create a Cloud ML Engine model:
    
        gcloud ml-engine models create $MODEL_NAME --regions=$REGION
    

3. Select the job output to use. The following sample uses the job named `census_dist_1`.
    
        OUTPUT_PATH=gs://$BUCKET_NAME/census_dist_1
    

4. Look up the full path of your exported trained model binaries:
    
        gsutil ls -r $OUTPUT_PATH/export
    

5. Find a directory named `$OUTPUT_PATH/export/census/` and copy this directory path (without the : at the end) and set the environment variable `MODEL_BINARIES` to its value. For example:
    
        MODEL_BINARIES=gs://$BUCKET_NAME/census_dist_1/export/census/<ID>/

Where `$BUCKET_NAME` is your Cloud Storage bucket name, and `census_dist_1` is the output directory.

6. Run the following command to create a version `v1`:
    
        gcloud ml-engine versions create v1  \
        --model $MODEL_NAME \
        --origin $MODEL_BINARIES \
        --runtime-version 1.10
    

You can get a list of your models using the `models list` command.
    
    
    gcloud ml-engine models list
    

## Send an online prediction request to a deployed model

You can now send prediction requests to your model. For example, the following command sends an online prediction request using a `test.json` file that you downloaded as part of the sample GitHub repository.
    
    
    gcloud ml-engine predict \
        --model $MODEL_NAME \
        --version v1 \
        --json-instances ../test.json
    

The response includes the probabilities of each label (**>50K** and **<=50K**) based on the data entry in `test.json`, thus indicating whether the predicted income is greater than or less than 50,000 dollars.

The response looks like this:
    
    
    CLASSES       PROBABILITIES
    [u'0', u'1']  [0.9969545602798462, 0.0030454816296696663]
    

### Submit a batch prediction job

The batch prediction service is useful if you have large amounts of data and no latency requirements on receiving prediction results. This uses the same format as online prediction, but requires data be stored in Cloud Storage.

1. Set a name for the job.
    
        JOB_NAME=census_prediction_1
    

2. Set the output path.
    
        OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
    

3. Submit the prediction job.
    
        gcloud ml-engine jobs submit prediction $JOB_NAME \
        --model $MODEL_NAME \
        --version v1 \
        --data-format TEXT \
        --region $REGION \
        --input-paths $TEST_JSON \
        --output-path $OUTPUT_PATH/predictions
    

Unlike the previous commands, this one returns immediately. Check the progress of the job and wait for it to finish:
    
    
    gcloud ml-engine jobs describe $JOB_NAME
    

You should see `state: SUCCEEDED` once the job completes; this may take several minutes. You can also see the job logs in your terminal using
    
    
    gcloud ml-engine jobs stream-logs $JOB_NAME
    

Alternatively, you can check the progress in **ML Engine** > **Jobs** on [GCP Console][31].

After the job succeeds, you can:

* Read the output summary.
    
        gsutil cat $OUTPUT_PATH/predictions/prediction.results-00000-of-00001
    

You should see output similar to the following.
    
        {"probabilities": [0.9962924122810364, 0.003707568161189556], "logits": [-5.593664646148682], "classes": 0, "logistic": [0.003707568161189556]}
    

* List the other files that the job produced using the `gsutil ls` command.
    
        gsutil ls -r $OUTPUT_PATH
    

Compared to online prediction, batch prediction:

* Is slower for this small number of instances (but is more suitable for large numbers of instances).
* May return output in a different order than the input (but the numeric index allows each output to be matched to its corresponding input instance; this is not necessary for online prediction since the outputs are returned in the same order as the original input instances).

After the predictions are available, the next step is usually to ingest these predictions into a database or data processing pipeline.

In this sample, you deployed the model before running the batch prediction, but it's possible to skip that step by specifying the model binaries URI when you submit the batch prediction job. One advantage of generating predictions from a model before deploying it is that you can evaluate the model's performance on different evaluation datasets to help you decide whether the model meets your criteria for deployment.

## Cleaning up

If you've finished analyzing the output from your training and prediction runs, you can avoid incurring additional charges to your GCP account for the Cloud Storage directories used in this guide:

1. Open a terminal window (if not already open).

2. Use the gsutil rm command with the -r flag to delete the directory that contains your most recent job:
    
        gsutil rm -r gs://$BUCKET_NAME/$JOB_NAME
    

If successful, the command returns a message similar to this:
    
    
    Removing gs://my-awesome-bucket/just-a-folder/cloud-storage.logo.png#1456530077282000...
    Removing gs://my-awesome-bucket/...

Repeat the command for any other directories that you created for this sample.

Alternatively, if you have no other data stored in the bucket, you can run the `gsutil rm -r` command on the bucket itself.

## About the data

The [Census Income Data Set][37] that this sample uses for training is hosted by the [UC Irvine Machine Learning Repository][38].

Census data courtesy of Lichman, M. (2013). UCI Machine Learning Repository . Irvine, CA: University of California, School of Information and Computer Science. This dataset is publicly available for anyone to use under the following terms provided by the Dataset Source -  \- and is provided "AS IS" without any warranty, express or implied, from Google. Google disclaims all liability for any damages, direct or indirect, resulting from the use of the dataset.

**Note:** The `adult.test.csv` file included in the GitHub repository has been modified from the original source data to remove an extraneous trailing period character at the end of each line.


[1]: https://research.googleblog.com/2016/06/wide-deep-learning-better-together-with.html
[2]: https://cloud.google.com/products/calculator/
[3]: https://accounts.google.com/Login
[4]: https://accounts.google.com/SignUp
[5]: https://console.cloud.google.com/cloud-resource-manager
[6]: https://cloud.google.com/billing/docs/how-to/modify-project
[7]: https://console.cloud.google.com/flows/enableapi?apiid=ml.googleapis.com,compute_component
[8]: https://console.cloud.google.com/apis/credentials/serviceaccountkey
[9]: https://console.cloud.google.com/
[10]: https://cloud.google.com/iam/docs/granting-roles-to-service-accounts
[11]: https://cloud.google.com/sdk/docs/
[12]: https://www.python.org/downloads/
[13]: https://pip.pypa.io/en/stable/installing/#installing-with-get-pip-py
[14]: https://pip.pypa.io/en/stable/installing/#upgrading-pip
[15]: https://virtualenv.pypa.io/en/stable/installation/
[16]: https://virtualenv.pypa.io/en/stable/userguide/
[17]: https://cloud.google.com/shell/docs/images/shell_icon.png
[18]: https://cloud.google.com/shell/docs/images/new-console.png
[19]: https://www.tensorflow.org/install/
[20]: https://www.tensorflow.org/install/install_mac#common_installation_problems
[21]: https://cloud.google.com/ml-engine/docs/tensorflow/versioning#set-python-version-training
[22]: https://github.com/GoogleCloudPlatform/cloudml-samples/archive/master.zip
[23]: https://cloud.google.com#about-data
[24]: https://www.tensorflow.org/install/install_mac#installing_with_docker
[25]: https://www.tensorflow.org/get_started/summaries_and_tensorboard
[26]: https://cloud.google.com/shell/docs/features#web_preview
[27]: https://cloud.google.com/ml-engine/docs/images/tensorboard-accuracy.png
[28]: https://cloud.google.com/ml-engine/docs/working-with-cloud-storage#setup-different-project
[29]: https://cloud.google.com/ml-engine/docs/regions
[30]: https://cloud.google.com/ml-engine/docs/tensorflow/machine-types
[31]: https://console.cloud.google.com/mlengine/jobs
[32]: https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction#inspect-local-output
[33]: https://cloud.google.com/ml-engine/docs/images/ml-engine-console-logs-accuracy.png
[34]: https://cloud.google.com/sdk/gcloud/reference/ml-engine/jobs/stream-logs
[35]: https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#scaletier
[36]: https://cloud.google.com/ml-engine/docs/tensorflow/hyperparameter-tuning-overview
[37]: https://archive.ics.uci.edu/ml/datasets/Census+Income
[38]: https://archive.ics.uci.edu/ml/datasets/

  