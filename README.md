# Fashion-MNIST Image Classification Using Fully Connected Neural Networks
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/robrj017/fashion-mnist-fcnn/blob/main/fashion_mnist_fcnn.ipynb)

## Results at a Glance

| Metric | Result |
|---|---:|
| Best Validation Accuracy | **90.67%** |
| Final Test Accuracy | **88.97%** |
| Test Loss | **0.3172** |
| Generalization Gap | **1.45 percentage points** |
| Expected Calibration Error | **0.0230** |
| Highly Confident Errors | **194 / 1,103** |

**Best Model:** Dropout-regularized FCNN (512 → 256 → 128 → 10)

The experiments show that Dropout provided a more meaningful improvement than simply increasing network depth, while the final analysis also examined error patterns, confidence, calibration, and generalization.

## About The Project

This project explores how well Fully Connected Neural Networks (FCNNs) can perform on the Fashion-MNIST image classification problem.

Instead of training just one model and reporting its accuracy, I followed a structured experimentation approach. I started with a simple baseline model and then experimented with a deeper network, Dropout, L2 regularization, and different learning rates.

The main goal was to understand how model complexity, regularization, and optimization affect performance and generalization.


## ABOUT THE DATASET

Fashion-MNIST is a dataset of 70,000 grayscale images of clothing and fashion items.

Each image is 28 × 28 pixels, grayscale, and belongs to one of 10 classes.

### The 10 classes are:
1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

The dataset contains 60,000 training images and 10,000 test images.


### WHAT I TRIED

The project was built as a series of controlled experiments.

1. Baseline Fcnn

I first created a simple Fully Connected Neural Network to establish a baseline.

28 × 28 Image
↓
Flatten
↓
Dense(128, ReLU)
↓
Dense(10, Softmax)

This model provides a reference point for the experiments that follow.

2. Deep Fcnn

I then increased the network capacity by adding more hidden layers.

28 × 28 Image
↓
Flatten
↓
Dense(512, ReLU)
↓
Dense(256, ReLU)
↓
Dense(128, ReLU)
↓
Dense(10, Softmax)

The purpose was to see whether a deeper network could learn more complex representations and improve classification performance.

3. Dropout Regularization

Dropout was added after each hidden layer using a dropout rate of 30%.

This experiment was designed to reduce overfitting and improve generalization.

4. L2 Regularization

I also tested L2 regularization with a coefficient of 0.0001 on the hidden-layer kernels.

This allowed me to compare two different approaches to controlling model complexity.

5. Learning Rate Experiment

Finally, I compared two learning rates using the Dropout architecture:
• 0.01
• 0.001

The architecture and regularization settings were kept unchanged so that the effect of the learning rate could be studied separately.


## MODEL SELECTION

The dataset was divided into training, validation, and test data.

The validation set was used to compare the different experiments and select the final model.

The test set was kept separate and was used only after model selection.

This prevents the test data from influencing the model-selection process.


## RESULTS

The current experiments produced the following validation results:

Baseline FCNN: 89.12%
Deep FCNN: 89.82%
Dropout FCNN: 90.67%
L2 FCNN: 89.78%
Dropout + Learning Rate 0.001: 87.43%

The Dropout model achieved the highest validation accuracy and was selected as the final model.

## Final Test Results

Test Accuracy: 88.97%
Test Loss: 0.3172
Generalization Gap: 1.45 percentage points
Expected Calibration Error: 0.0230
Incorrect Predictions: 1,103
Highly Confident Errors: 194
High-Confidence Error Rate: 17.59%


## WHAT THE RESULTS SHOW

One of the main observations from the experiments is that simply making the network deeper did not provide a very large improvement.

The baseline achieved 89.12% validation accuracy, while the deeper FCNN reached 89.82%.

Adding Dropout, however, improved validation performance to 90.67%.

This suggests that controlling overfitting was more useful in this experiment than simply increasing the capacity of the network.

The lower learning rate of 0.001 also performed worse than the original 0.01 configuration.

The selected Dropout model achieved 88.97% accuracy on the unseen test set, giving a generalization gap of approximately 1.45 percentage points.


## ERROR ANALYSIS

I did not stop at accuracy. The final model was also analyzed in more detail.

A confusion matrix was used to understand which Fashion-MNIST classes were most frequently confused with one another.

Precision, recall, F1-score, and support were examined for each class.

The confusion matrix was further analyzed to identify the class pairs responsible for the largest number of incorrect predictions.

The error rate for each class was also calculated to identify categories that were more difficult for the model.


## PREDICTION CONFIDENCE

The model's prediction confidence was also analyzed.

Out of 1,103 incorrect test predictions, 194 were made with at least 90% confidence.

That means the model can sometimes be very confident even when it is wrong.

This is an important observation because accuracy alone does not tell us how reliable a model's confidence estimates are.


## CONFIDENCE CALIBRATION

To investigate prediction reliability, I used a reliability diagram and Expected Calibration Error (ECE).

The final model achieved:

Ece = 0.0230

A lower ECE generally indicates that predicted confidence is closer to the model's observed accuracy across confidence ranges.

The calibration analysis therefore provides another perspective on model reliability beyond classification accuracy.


## TRAINING AND OVERFITTING ANALYSIS

Training and validation accuracy and loss were tracked during training.

These curves were used to examine learning behavior, validation performance, overfitting, generalization, and the effect of early stopping.

Early stopping was used to restore the best model weights based on validation loss.


## LIMITATIONS OF THE APPROACH

Although the FCNN performs well on Fashion-MNIST, fully connected networks are not specifically designed for image data.

The images are flattened into a one-dimensional vector, so the network does not explicitly take advantage of local spatial relationships between neighboring pixels.

Fully connected architectures can also become parameter-heavy as image resolution and network size increase.

For more complex image-classification problems, Convolutional Neural Networks (CNNs) are generally a more suitable architecture because they can learn local spatial patterns such as edges, textures, and shapes.


TECHNOLOGIES USED

• Python
• TensorFlow / Keras
• NumPy
• Matplotlib
• Seaborn
• Scikit-learn
• Google Colab


## PROJECT STRUCTURE

fashion-mnist-fcnn/
│
├── fashion_mnist_fcnn.ipynb
└── README.md

The notebook contains the complete workflow, including data exploration, preprocessing, model development, experimentation, evaluation, error analysis, and inference.


## HOW TO RUN

The project was developed in Google Colab.

Open fashion_mnist_fcnn.ipynb in Google Colab and run the notebook from beginning to end.

The notebook automatically loads the Fashion-MNIST dataset using TensorFlow/Keras.

For a local environment, the main dependencies can be installed with:

pip install tensorflow numpy matplotlib seaborn scikit-learn


## FUTURE IMPROVEMENTS

There are several ways this project could be extended:

• Compare the FCNN with a Convolutional Neural Network
• Experiment with Batch Normalization
• Test different optimizers
• Perform more systematic hyperparameter tuning
• Investigate per-class calibration
• Compare parameter efficiency between FCNN and CNN models
• Deploy the final model as a small web application


## CONCLUSION

This project was designed to understand not just how accurate a Fully Connected Neural Network can be on Fashion-MNIST, but also how different modeling decisions affect its behavior.

The experiments showed that Dropout provided the strongest validation performance among the tested configurations, while simply increasing network depth produced only a modest improvement.

The final selected model achieved 88.97% accuracy on the unseen test set.

The additional analysis of errors, prediction confidence, calibration, and training dynamics provides a more complete picture of the model than accuracy alone.

At the same time, the limitations of FCNNs for image data make this project a useful starting point for a future comparison with Convolutional Neural Networks.


AUTHOR

Ritwik Raj

Data Science / Machine Learning Portfolio Project
