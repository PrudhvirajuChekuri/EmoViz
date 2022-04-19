# EmoViz(Facial Emotion Detector) 🙂😀😮😤😒😔😨

https://user-images.githubusercontent.com/96725900/164004710-2fbaaa02-fe4f-4645-b08d-d1426139ced5.mp4

![7Emotions](https://user-images.githubusercontent.com/96725900/163942354-e7befdcc-1772-4877-b271-bab3d5c9eabf.png)

# 📝Description

Facial Emotion Detection is one of the useful and toughest Machine Learning tasks because of the intra-class variation in expressions among the people. The best usecase of FED is in human-machine interaction. EmoViz is a facial emotion detection system built using TensorFlow, which takes image input and display one of the seven emotions(Neutral, Happy, Surprise, Angry, Disgust, Sad, Fear).

# 🤖About the Models

The system employs two models: one for face detection and another for facial emotion classification.

## Face Detector

* Face detection model is built using TensorFlow 2.0 object detection API, where I used SSD MOBILENET V2 as the pretrained model.
* To built a face detection model first get the data(There will be many resources out there or else take some random pictures from internet and the number is dependent, in my case it's 50 for training and 10 for validation) to train the model.
* And then follow [this](https://tensorflow-object-detection-api-tutorial.readthedocs.io/) documentation to train a face detector on the gathered data. If u want a video tutorial check [this](https://youtu.be/XoMiveY_1Z4).
* I'd used colab to train the model.

## Facial Emotion Classifier

* Facial Emotion Classification model is built using TensorFlow 2.0, where I trained a deep convolutional neural network on RAF-DB. It's a generalised modelwith a training accuracy of 90% and test accuracy of 83%.
*  I'll create a separate repo on how to create a model like this, so if you're here then check my repos.
*  RAF-DB is a private database and to download it you need to send an email as shown on their [website](http://www.whdeng.cn/raf/model1.html#:~:text=Real%2Dworld%20Affective%20Faces%20Database%20(RAF%2DDB)%20is,labeled%20by%20about%2040%20annotators.).
*  First I'd started training the model on my local machine(GPU - GTX 1660ti), but when I started incresing the complexity of the model it took a lot of time to train and also started crying like a baby. 
*  I can't experiment fastly, So I'had shifted to kaggle to train the model. I'had choosen kaggle over colab because of it's more gpu time and resources and it made experimentation faster.

## How it works

The input image is first given to the face detector to detect the face and then this is given to the facial emotion classifier to classify the emotion.

# 🖥Installation

## 🛠Requirements

* Python 3.7+ 
* Other requirement are mentioned in requirements.txt and can be installed as shown below.

## ⚙️Setup
```
pip install -r requirements.txt
```
That's it you're set to go.
# 🎯Inference
```
python app.py
```

# License
Distributed under the MIT License. See LICENSE.txt for more information.

# Credits
* http://www.whdeng.cn/raf/li_RAFDB_2017_CVPR.pdf
