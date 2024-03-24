# HPV_classification
Developing a deep learning model, utilizing Convolutional Neural Networks (CNNs) and HPV dataset to accurately distinguish between tumour and non-tumour blood cell samples

## Overview
This project is focused on creating a proof of concept for classifying blood cell images using advanced machine-learning techniques. We have employed transfer learning on pre-trained networks to identify key features in blood cell images, comparing the performance of traditional models against those utilizing expert-derived masks.

## Dataset
The dataset contains blood cell images from an outdated instrument. We've divided this dataset into separate training, validation, and test sets. Initial experiments were performed using raw images, which were followed by runs using masked images for enhanced feature extraction.
![download](https://github.com/bonitavw/HPV_classification/assets/45999599/2a0d4be5-151d-48f2-837a-df2d8f2ab465)

## Methods
### Preprocessing and Data Splitting
Unzipped and loaded the dataset into the analysis environment.
Split the images into training (70%), validation (15%), and test (15%) sets, ensuring there's no data leakage across these splits.

### Model Development
Two neural networks, VGG16 and ResNet50, were trained using transfer learning. Both models underwent specific pre-processing to optimize the training, including colour space adjustments and normalization. The models were trained for 20 epochs with a batch size of 17 to prevent overfitting and make the most of the limited dataset.

A novel approach was taken where masks provided by domain experts were applied to the training and validation images to guide the network's attention to regions of interest.

Model Improvement Techniques:
- Data Augmentation: Applied to combat the limitations of a small dataset.
- Early Stopping and Model Checkpointing: To cease training when validation performance degrades.
- Regularization: Attempted to reduce overfitting through L1/L2 penalties on layer parameters.


## Conclusion
Data augmentation and early stopping combined with best model saving, seem to work in our data. We don't see much improvement with regularisation.

Data augmentation works very well since it improves the quality of the dataset, which is probably the biggest impediment in this project. Since we are working with a really small dataset (299 samples), from which 50% of the samples are represented by one class only, it is almost impossible to manage to train a model adequately. Using data augmentation, we managed to balance the datasets as well as increment the number of samples to train. Still, it can be a dangerous tool to use, because when applied to small datasets, we can end up misrepresenting certain characteristics. Early stopping combined with storing the best model also seems to work, because we manage both, to make the training process more efficient, as well as reduce overfitting. The regularisation probably didn't work too well, because we already use other techniques with the same purpose.

During this project, we also noticed that the script provided by the labs, to split the data into train, validation and test was incorrect, since it allows for overlap between the train data and the test. This explains why all our models perform incredibly well with the test dataset, but very poorly with the validation dataset, because the model is just overfitting the train data, and then, since it overlaps with the test data, it doesn't require much abstraction. To solve this problem, we have implemented a different notebook where we solved the error of the provided script so that the 3 data frames are not overlapping.
