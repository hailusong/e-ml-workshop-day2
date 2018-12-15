# Creating an Object Detection Application Using TensorFlow

## TensorFlow architecture overview

The object detection application uses the following components:

* [TensorFlow][1]. An open source machine learning library developed by researchers and engineers within Google's Machine Intelligence research organization. TensorFlow runs on multiple computers to distribute the training workloads.
* [Object Detection API][2]. An open source framework built on top of TensorFlow that makes it easy to construct, train, and deploy object detection models.
* [Pre-trained object detection models][3]. The Object Detection API provides pre-trained object detection models for users running inference jobs. Users are not required to train models from scratch.

### Local implementation

The following diagram shows how this tutorial is implemented. The web application is deployed to a VM instance running on Compute Engine.

![image][4]

When the client uploads an image to the application, the application runs the inference job locally. The pre-trained model returns the labels of detected objects, and the image coordinates of the corresponding objects. Using these values, the application generates new images populated with rectangles around the detected objects. Separate images are generated for each object category, allowing the client to discriminate between selected objects.

### Remote implementation

You can deploy the pre-trained model on [Google Cloud Machine Learning Engine][5] to provide an API service for inference. If you do, the web application sends an API request to detect objects in the uploaded image, instead of running the inference job locally.

TensorFlow allows you to choose which platform to execute inference jobs on depending on your business needs. This flexibility shows the advantage of Google Cloud Platform and TensorFlow as an open platform for machine learning.

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

![image][8]

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

[1]: https://www.tensorflow.org/
[2]: https://github.com/tensorflow/models/tree/master/research/object_detection
[3]: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
[4]: https://cloud.google.com/solutions/images/object-detection-tensorflow-architecture.svg
[5]: https://cloud.google.com/ml-engine
[6]: http://mscoco.org/
[7]: https://console.cloud.google.com/compute/instances
[8]: https://cloud.google.com/solutions/images/object-detection-tensorflow-example.png