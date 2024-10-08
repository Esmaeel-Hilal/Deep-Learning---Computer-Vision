# Developing a Food Image Classification Model with Two-Stage Transfer Learning Using EfficientNet
I have finished this project as part of my journey to pursue my passion for AI. This project is the first milestone project in the “TensorFlow for Deep Learning Bootcamp” course on Udemy. While the course instructor offered guidance, I completed the project independently.

The goal was to train a multiclass classification model that surpasses the Deepfood model. The Deepfood model used a convolutional neural network trained for **2-3 days and achieved a top-1 accuracy of 77.4%** on the food101 dataset.
To achieve this goal, I used EfficientNet (B0 version) as the backbone model, added a global average pooling layer, batch normalization, and a dropout layer. I trained the model in two phases as described in the “Two-Stage Transfer Learning” section and achieved a **top-1 accuracy of 80.66%**. This is a close result compared to **the current state-of-the-art accuracy of 83.35%.** (For reference to the state-of-the-art results, please see the paper listed as resource number 6 in the 'Resources' section).  

The training process took approximately **6 hours and 30 minutes** and it was completed on my personal laptop, which is equipped with an Nvidia 1050 GTX.

I deployed the project using **Flask Web Framework**. (A link for the deployment tutorial is provided in the resources section)

**Key Concepts:** Deep Learning, Convolutional Neural Networks, Computer Vision, Multi-class Classification, Transfer Learning, Feature Extraction, Fine-tuning, EfficientNet. 

## Data loading and preparation:
After downloading and unzipping the dataset. I prepared the data for training by:
- Resizing the images (the images in food101 are of different sizes).
- Batching the images for better training speed.
- Splitting the data into 75% for training and 25% for testing according to the dataset paper.
- Shuffling the training dataset only (the testing dataset was not shuffled for further outcomes analysis).
- Converting the data type to “float32” (the dataset datatype is “unit8”).

Note: These steps can be accomplished using a single method in the TensorFlow framework.

## Two-Stage Transfer Learning:
In this project, transfer learning was applied to enhance the model's performance, following a two-stage approach.

The Transfer Learning was performed in two steps: 
1. Feature Extraction: by freezing the backbone model and training the entire model for 15 epochs.
2. Fine-Tuning: by unfreezing the backbone and continuing training for 10 more epochs, following the approach proposed in the ULMFit paper.

Note:
Based on a Keras tutorial on Image Classification via Fine-Tuning with EfficientNet (link provided in the resources section):
- **The BatchNormalization layers must remain frozen during unfreezing**. If set to trainable, the accuracy of the first epoch after unfreezing will drop significantly.
- **Each block must be turned on or off in its entirety** because the architecture includes a shortcut from the first layer to the last in each block. Disabling parts of a block can significantly degrade the model's performance.

This adjustment was crucial. After applying these guidelines, my model surpassed the performance of the DeepFood model. Prior to following this note, **the model couldn’t exceed 76% accuracy**.

### Stage 1 – Feature Extraction:  
- EfficientNetB0 was utilized as a feature extraction layer by excluding its top layer and freezing all the other layers to prevent weight modification.
- A dense layer with 101 neurons and a "softmax" activation function was used as the output layer.
- The model was saved using the checkpoint callback for future fine-tuning or model restoration.

After training the model for only 15 epochs, which took about 2 hours, the model achieved **an accuracy of 73.73%** on the test set. 

### Stage 2 – Fine-tuning:
EfficientNetB0's layers were unfrozen for fine-tuning (except the BatchNormalization layers). 

After training the model for another 10 epochs, it took approximately 4 hours and 30 minutes. 
The model achieved **an accuracy of 80.66% on the test set**, surpassing the Deepfood model’s accuracy. 

**Mission accomplished!**

## Analyzing the outcomes:
- To evaluate the model’s results, I used the following metrics for each class:
    - Precision.
    - Recall.
    - F1-Score.
- A confusion matrix was used to comprehensively analyze the model's predictions.

After conducting further analysis, I identified the classes that were classified with high probability but were wrong. I used a Pandas data frame to arrange and sort the outcomes, aiming to explore why the model was confident that these images belonged to a certain class when they actually didn't. Upon inspection, I discovered that these misclassified images were very similar in almost every aspect to the wrongly predicted class. It was difficult even for a human, including myself, to confidently classify these images.

In conclusion, despite being trained for a significantly shorter time than the Deepfood model, my model produced better results. This serves as further evidence of the rapid evolution of deep learning, both in software and the hardware necessary for training. The Deepfood model was developed in 2016, while EfficientNetBX, the primary component of my model, was developed in 2020. The four-year gap between the development of these models played a significant role in the final results. Additionally, hardware advancements contributed to faster training. The total training time for my model took 6 hours and 30 minutes, compared to the 2-3 days required for training the Deepfood model.

## Resources:
1. Food101 dataset: [Food 101] (https://www.kaggle.com/datasets/dansbecker/food-101) 
2. EfficientNet paper: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
3. Deepfood paper: [DeepFood: Deep Learning-Based Food Image Recognition for Computer-Aided Dietary Assessment](https://arxiv.org/abs/1606.05675)
4. ULMFit approach: [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146)
5. TensorFlow for Deep Learning Bootcamp Course: [TensorFlow for Deep Learning Bootcamp | Udemy](https://www.udemy.com/course/tensorflow-developer-certificate-machine-learning-zero-to-mastery/?couponCode=24T6MT62024)
6. Deployment Tutorial: [Deploy Machine Learning Models using Flask](https://youtu.be/0nr6TPKlrN0?si=sXaa2Unorxr_wZ89)
7. [Image classification via fine-tuning with EfficientNet](https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/)
8. [Optimizing Food101 Classification with Transfer Learning: A Fine-Tuning Approach Using EfficientNetB0](https://sciencepublishinggroup.com/article/10.11648/j.ijiis.20241304.11)
