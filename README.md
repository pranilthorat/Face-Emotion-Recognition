# Live Class Monitoring System(Face Emotion Recognition)
## Introduction

Face Emotion recognition is the process of identifying human emotion. People vary widely in their accuracy at recognizing the emotions of others. Use of technology to help people with emotion recognition is a relatively nascent research area. Generally, the technology works best if it uses multiple modalities in context. To date, the most work has been conducted on automating the recognition of facial expressions from video, spoken expressions from audio, written expressions from text, and physiology as measured by wearables.

Facial expressions are a form of nonverbal communication. Various studies have been done for the classification of these facial expressions. There is strong evidence for the universal facial expressions of seven emotions which include: neutral happy, sadness, anger, disgust, fear, and surprise. So it is very important to detect these emotions on the face as it has wide applications in the field of Computer Vision and Artificial Intelligence. These fields are researching on the facial emotions to get the sentiments of the humans automatically.

![image](https://user-images.githubusercontent.com/92014177/155081533-a356d549-087e-4387-9945-a2c67aa10dbf.png)


## Problem Statements

The Indian education landscape has been undergoing rapid changes for the past 10 years owing to the advancement of web-based learning services, specifically, eLearning platforms.

Global E-learning is estimated to witness an 8X over the next 5 years to reach USD 2B in 2021. India is expected to grow with a CAGR of 44% crossing the 10M users mark in 2021. Although the market is growing on a rapid scale, there are major challenges associated with digital learning when compared with brick and mortar classrooms. One of many challenges is how to ensure quality learning for students. Digital platforms might overpower physical classrooms in terms of content quality but when it comes to understanding whether students are able to grasp the content in a live class scenario is yet an open-end challenge. In a physical classroom during a lecturing teacher can see the faces and assess the emotion of the class and tune their lecture accordingly, whether he is going fast or slow. He can identify students who need special attention.

Digital classrooms are conducted via video telephony software program (ex-Zoom) where it’s not possible for medium scale class (25-50) to see all students and access the mood. Because of this drawback, students are not focusing on content due to lack of surveillance.

While digital platforms have limitations in terms of physical surveillance but it comes with the power of data and machines which can work for you. It provides data in the form of video, audio, and texts which can be analyzed using deep learning algorithms.

Deep learning backed system not only solves the surveillance issue, but it also removes the human bias from the system, and all information is no longer in the teacher’s brain rather translated in numbers that can be analyzed and tracked.

I will solve the above-mentioned challenge by applying deep learning algorithms to live video data. The solution to this problem is by recognizing facial emotions.

## What is Face Emotion Recognition?

- Facial emotion recognition is the process of detecting human emotions from facial expressions.
- The human brain recognizes emotions automatically, and software has now been developed that can recognize emotions as well.
- This is a few shot learning live face emotion detection system.
- The model should be able to real-time identify the emotions of students in a live class.

## Head-start References

❖ https://towardsdatascience.com/face-detection-recognition-and-emotion-detection-in-8-lines-of-code-b2ce32d4d5de

❖ https://towardsdatascience.com/video-facial-expression-detection-with-deep-learning-applying-fast-ai-d9dcfd5bcf10

❖ https://github.com/atulapra/Emotion-detection

❖ https://medium.com/analytics-vidhya/building-a-real-time-emotion-detector-towards-machine-with-e-q-c20b17f89220

## Dataset link 
❖ https://www.kaggle.com/deadskull7/fer2013

## Dataset Information

The data comes from the past Kaggle competition “Challenges in Representation Learning: Facial Expression Recognition Challenge”. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image.

This dataset contains 35887 grayscale 48x48 pixel face images.

Each image corresponds to a facial expression in one of seven categories

Labels:

0 - Angry 😠

1 - Disgust 😧

2 - Fear 😨

3 - Happy 😃

4 - Sad 😞

5 - Surprise 😮

6 - Neutral 😐


## Project Approch

**Step 1. Build Model**

We have used **Five different models**  as follows:

    Model 1- Mobilenet Model
    Model 2- Dexpression Model
    Model 3- CNN Model
    Model 4- Densenet Model
    Model 5- Resnet Mode 



**Step 2. Real Time Prediction**

And then we perform **Real Time Prediction** on our best model using webcam on Google colab itself.

      - Run webcam on Google Colab
      - Load our best Model
      - Real Time prediction

**Step 3. Deployment**

And lastly we have **deployed** it on **Amazon WEB Services (AWS)**

# Build Model:-

## 1) Mobilenet:
MobileNet is an efficient and portable CNN architecture that is used in real-world applications. MobileNets primarily use depth-separable convolutions in place of the standard convolutions used in earlier architectures to build lighter models. MobileNets introduces two new global hyperparameters (width multiplier and resolution multiplier) that enable model developers to trade off latency or accuracy for speed and low size based on their needs.
 
![image](https://user-images.githubusercontent.com/92014177/155086757-9a04329f-dc8e-42e7-95c9-6694f2c1957f.png)

 
## 2) Dexpression:
The suggested architecture outperforms the current state of the art utilizing CNNs by 99.6 percent for CKP and 98.63 percent for MMI. Face recognition software has a wide range of applications, including human-computer interface and safety systems. This is because nonverbal cues are vital types of communication that play an important part in interpersonal interactions. The usefulness and dependability of the suggested work for real-world applications is supported by the performance of the proposed architecture.

![image](https://user-images.githubusercontent.com/92014177/155086797-e1ecf4f5-dfb6-492b-b5da-99bff69c707e.png)


 
## 3) CNN:
Basic CNN architecture details:
• Input layer - The input layer in CNN should contain image data.
• Convo layer - The convo layer is sometimes called the feature extractor layer because features of the image are get extracted within this layer 
• Pooling layer - Pooling is used to reduce the dimensionality of each feature while retaining the most important information. It is used between two convolution layers.
• Fully CL - Fully connected layer involves weights, biases, and neurons. It connects neurons in one layer to neurons in another layer. It is used to classify images between different categories by training and placed before the output layer
• Output Layer - The output layer contains the label which is in the form of a one-hot encoded.
A Convolutional Neural Network (CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. 
 
![image](https://user-images.githubusercontent.com/92014177/155086845-1cc6d324-9d7c-41a3-8b20-47b1f7078e47.png)

## 4) Densenet:
DenseNet was developed specifically to improve the declining accuracy caused by the vanishing gradient in high-level neural networks. In simpler terms, due to the longer path between the input layer and the output layer, the information vanishes before reaching its destination. 

 ![image](https://user-images.githubusercontent.com/92014177/155086885-91d356e8-051c-4c9d-bde4-9991f5a862de.png)

## 5) ResNet:
The term micro-architecture refers to the set of “building blocks” used to construct the network. A collection of micro-architecture building blocks (along with your standard CONV, POOL, etc. layers) leads to the macro-architecture (i.e., the end network itself). First introduced by He et al. in their 2015 paper, the ResNet architecture has become a seminal work, demonstrating that extremely deep networks can be trained using standard SGD (and a reasonable initialization function) through the use of residual modules:

Further accuracy can be obtained by updating the residual module to use identity mappings, as demonstrated in their 2016 follow-up publication,
 Identity Mappings in Deep Residual Networks:

That said, keep in mind that the ResNet50 (as in 50 weight layers) implementation in the Keras core is based on the former 2015 paper.
Even though ResNet is much deeper than VGG16 and VGG19, the model size is actually substantially smaller due to the usage of global average pooling rather than fully-connected layers — this reduces the model size down to 102MB for ResNet50.

![image](https://user-images.githubusercontent.com/92014177/155086937-2d432737-0b7a-4038-a17a-411fb0a73004.png)

# Model performance:-

# i) Confusion Matrix-
The confusion matrix is a table that summarizes how successful the classification model is at predicting examples belonging to various classes. One axis of the confusion matrix is the label that the model predicted, and the other axis is the actual label.

Precision, Recall, F1 score and Support-
Precision is the ratio of correct positive predictions to the overall number of positive predictions: TP/TP+FP
Recall is the ratio of correct positive predictions to the overall number of positive examples in the set: TP/FN+TP
It is also called the F Score or the F Measure. Put another way, the F1 score conveys the balance between the precision and the recall The F1 Score is the 2*((precision*recall)/(precision+recall))
Support is the number of actual occurrences of the class in the specified dataset. Imbalanced support in the training data may indicate structural weaknesses in the reported scores of the classifier and could indicate the need for stratified sampling or rebalancing

![image](https://user-images.githubusercontent.com/92014177/155087452-eb552003-82a3-48e8-9ffe-069d17a62bf0.png)
![image](https://user-images.githubusercontent.com/92014177/155087649-f590fcd5-8708-40d8-96b0-3a28e7abfb76.png)
![image](https://user-images.githubusercontent.com/92014177/155087676-547d4544-934b-4398-8bfc-32860869fbd8.png)
![image](https://user-images.githubusercontent.com/92014177/155087705-c0396374-dc02-4487-9285-3c79796a8ae7.png)
![image](https://user-images.githubusercontent.com/92014177/155087741-692dc6a5-77a9-4ab2-bc97-c0b62a4d474d.png)



# ii) Accuracy and loss curve-

Accuracy is a method for measuring a classification model's performance. It is typically expressed as a percentage. ... Accuracy is often graphed and monitored during the training phase though the value is often associated with the overall or final model accuracy. Accuracy is easier to interpret than loss.
Loss value implies how poorly or well a model behaves after each iteration of optimization. An accuracy metric is used to measure the algorithm's performance in an interpretable way. It is the measure of how accurate your model's prediction is compared to the true data.

![image](https://user-images.githubusercontent.com/92014177/155087342-1f467ea3-f07a-4d1d-9d0e-621a205eddd6.png)

# Real Time Face Emotion Detection:-
![image](https://user-images.githubusercontent.com/92014177/155088427-e246f0d4-4a8b-438b-a809-6f9a9c4b1eab.png)
![image](https://user-images.githubusercontent.com/92014177/155088477-40b99efc-ec6f-4fcc-beac-e29b40ad83ca.png)
![image](https://user-images.githubusercontent.com/92014177/155088527-4ce3b719-89a5-4d0c-8779-5ca70f6463d4.png)
![image](https://user-images.githubusercontent.com/92014177/155088564-035caedc-40f6-49b0-99e6-2819a1069664.png)


# Model Deployment:-

### Creating Web App Using Streamlit-
Streamlit is a Python framework for developing machine learning and data science web apps that is open-source. Using Streamlit, we can quickly create web apps and deploy them. You can use Streamlit to make an app the same way you'd make a Python programme. It's possible with Streamlit. Working on the interactive loop of coding and viewing results is a pleasure. In the web application.


### Deployment in cloud platform-
AWS (Amazon Web Services) is a comprehensive, evolving cloud computing platform provided by Amazon that includes a mixture of infrastructure as a service (IaaS), platform as a service (PaaS), and packaged software as a service (SaaS) offerings.


Deployment Link for AWS-

http://15.206.194.193:8501/

# Conclusion:

* All the models such as  Mobilenet, Dexpression, CNN, Densenet, and ResNet were evaluated.
* The ResNet model was chosen because it had the highest training accuracy of all the models, and its validation accuracy was nearly 72 percent, which is comparable to CNN models.
* As a result, we save this resnet model and use it to predict facial expressions.
* Since, the emotion counts of disgust and surprise images are less therefore on local webcam it hardly detect those emotions.
* Using streamlit, a front-end model was successfully created and ran on a local webserver.The Streamlit web application has been deployed on Amazon's AWS cloud platform.
* It was an amazing and fascinating project. This has taught me a lot.

# Ppt Presentation:-https://drive.google.com/file/d/1lSIBkvRnB0oUNUkR2A_qvyjOtgmHq64o/view?usp=sharing
# Video Presentation:-https://drive.google.com/file/d/1L_Uf6x8bCIHiGFdBMJn_9Ln8z6F0AvWt/view?usp=sharing
# Local Webcam Test (Demo Video):-https://drive.google.com/file/d/1fax5Y7QeZiUgkVj-FREiMy7Sykq_w_Ll/view?usp=sharing
