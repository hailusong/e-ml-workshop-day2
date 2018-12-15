# Machine Learn on GCP - Day 2

# ML Engine

## GCP ML Engine

###What is CloudML
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

![image][2]  ![image][3]  ![image][4]  


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

### Check Your Understanding

### Pre-trained models

You can use five pre-trained models with the Object Detection API. They are trained with [the COCO dataset][6] and are capable of detecting general objects in 80 categories.

The **COCO mAP** column shows the model's accuracy index. Higher numbers indicate better accuracy. As speed increases, accuracy decreases.

## Launch a VM instance

1. In the GCP Console, go to the VM Instances page. 

[Go to the VM Instances page][7]

2. Click **Create instance**. 
3. Set **Machine type** to **8 vCPUs**.
4. Click the **Customize** link next in the **Machine type** section.
5. In the **Memory** section, replace **30** with **8**. 
6. In the **Firewall** section, select **Allow HTTP traffic**. 
7. Click the **Management, security, disks, networking, sole tenancy** link, then click the **Networking** tab. 
8. Click the pencil icon next to the **default** row in the **Network interfaces** section.
9. Select **Create IP address** from the **External IP** dropdown to assign a static IP address. Input **staticip** in the **Name** field, then click **Reserve**.
10. Click **Create** to create the instance. 

## SSH into the instance

1. Click **SSH** to the right of the instance name to open an SSH terminal.

2. Enter the following command to switch to the root user:
    
        sudo -i
    

## Install the Object Detection API library

1. Install the prerequisite packages.
    
        apt-get update
    
        apt-get install -y protobuf-compiler python-pil python-lxml python-pip python-dev git
    
        pip install Flask==0.12.2 WTForms==2.1 Flask_WTF==0.14.2 Werkzeug==0.12.2
    
        pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp27-none-linux_x86_64.whl

2. Install the Object Detection API library.
    
        cd /opt
    
        git clone https://github.com/tensorflow/models
    
        cd models/research
    
        protoc object_detection/protos/*.proto --python_out=.

3. Download the pre-trained model binaries by running the following commands.
    
        mkdir -p /opt/graph_def
    
        cd /tmp
    
        for model in 
      ssd_mobilenet_v1_coco_11_06_2017 
      ssd_inception_v2_coco_11_06_2017 
      rfcn_resnet101_coco_11_06_2017 
      faster_rcnn_resnet101_coco_11_06_2017 
      faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017
    do 
      curl -OL http://download.tensorflow.org/models/object_detection/$model.tar.gz
      tar -xzf $model.tar.gz $model/frozen_inference_graph.pb
      cp -a $model /opt/graph_def/
    done

4. Choose a model for the web application to use. For example, to select `faster_rcnn_resnet101_coco_11_06_2017`, enter the following command:
    
        ln -sf /opt/graph_def/faster_rcnn_resnet101_coco_11_06_2017/frozen_inference_graph.pb /opt/graph_def/frozen_inference_graph.pb
    

## Install and launch the web application

1. Install the application.
    
        cd $HOME
    git clone https://github.com/GoogleCloudPlatform/tensorflow-object-detection-example
    cp -a tensorflow-object-detection-example/object_detection_app /opt/
    cp /opt/object_detection_app/object-detection.service /etc/systemd/system/
    systemctl daemon-reload
    

2. The application provides a simple user authentication mechanism. You can change the username and password by modifying the `/opt/object_detection_app/decorator.py` file.
    
        USERNAME = 'username'
    PASSWORD = 'passw0rd'

3. Launch the application.
    
        systemctl enable object-detection
    systemctl start object-detection
    systemctl status object-detection
    

The last command outputs the application status, as in the following example:
    
        ● object-detection.service - Object Detection API Demo
       Loaded: loaded (/opt/object_detection_app/object-detection.service; linked)
       Active: active (running) since Wed 2017-06-21 05:34:10 UTC; 22s ago
      Process: 7122 ExecStop=/bin/kill -TERM $MAINPID (code=exited, status=0/SUCCESS)
     Main PID: 7125 (app.py)
       CGroup: /system.slice/object-detection.service
               └─7125 /usr/bin/python /opt/object_detection_app/app.py  
    Jun 21 05:34:10 object-detection systemd[1]: Started Object Detection API Demo.
    Jun 21 05:34:26 object-detection app.py[7125]: 2017-06-2105:34:26.518736: W tensorflow/core/platform/cpu_fe...ons.
    Jun 21 05:34:26 object-detection app.py[7125]: 2017-06-2105:34:26.518790: W tensorflow/core/platform/cpu_fe...ons.
    Jun 21 05:34:26 object-detection app.py[7125]: 2017-06-2105:34:26.518795: W tensorflow/core/platform/cpu_fe...ons.
    Jun 21 05:34:26 object-detection app.py[7125]: * Running on http://0.0.0.0:80/ (Press CTRL+C to quit)
    Hint: Some lines were ellipsized, use -l to show in full.
    

The application loads the model binary immediately after launch. It will take a minute to start serving requests from clients. You'll see the message `Running on http://0.0.0.0:80/ (Press CTRL+C to quit)` when it's ready.

## Test the web application

Using a web browser, access the static IP address that was assigned when you launched the VM instance. When you upload an image file with a **JPEG**, **JPG**, or **PNG** extension, the application shows the result of the object detection inference, as shown in the following image. The inference might take up to 30 seconds, depending on the image.

![image][1]

The object names detected by the model are shown to the right of the image, in the application window. Click an object name to display rectangles surrounding the corresponding objects in the image. The rectangle thickness increases with object identification confidence.

In the above image, "fork", "cup", "dining table", "person", and "knife", are detected. After clicking **cup**, rectangles display around all detected cups in the image. Click **original** to see the original image. Test this model's accuracy by uploading images that contain different types of objects.

### Change the inference model

The following commands show how to change the inference model.
    
    
    systemctl stop object-detection
    ln -sf /opt/graph_def/[MODEL NAME]/frozen_inference_graph.pb /opt/graph_def/frozen_inference_graph.pb
    systemctl start object-detection
    

Replace [MODEL NAME] with one of the following options.

* `ssd_mobilenet_v1_coco_11_06_2017`
* `ssd_inception_v2_coco_11_06_2017`
* `rfcn_resnet101_coco_11_06_2017`
* `faster_rcnn_resnet101_coco_11_06_2017`
* `faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017`

[1]: https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/S1.png?raw=true

[2]: https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/tensorflow-logo.png?raw=true

[3]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/keras-logo.png?raw=true

[4]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/scikit-learn-logo.png?raw=true

[5]:https://github.com/ArctiqTeam/e-ml-workshop-
day2/blob/master/deck/images/xgboost-logo.png?raw=true
  
[6]:https://github.com/ArctiqTeam/e-ml-workshop-day2/blob/master/deck/images/refresher.png?raw=true