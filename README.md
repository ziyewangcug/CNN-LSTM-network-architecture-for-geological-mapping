**Abstract**

Application of convolutional recurrent neural networks for geological feature mapping based on geochemical exploration data.

-   This python code uses cross validation to tune the model and to find the optimal hyperparameters.
-   The input geochemical data include training and testing samples, where training samples are split into positive training data and negative training data, formatted as the example file.
-   The output result also formatted as a two-dimensional matrix including the locations of samples and the corresponding probability of target geological feature occurrence.

**Environment**

-   Python 3.7
-   TensorFlow 2.1
-   Keras 1.21
-   Numpy 1.15.0
-   Pandas 1.1

**Input dataset (example)**

-   positive training_data.csv

positive training data

-   negative training_data.csv

    negative training data

-   testing_data.csv

    testing data

-   testing_labels.csv

    testing labels

**Output dataset**

-   loss.csv

    loss function vs. epochs of model training

-   acc.csv

    accuracy vs. epochs of model training

-   result.csv

    the average predicted probability

Usage

-   set parameters: leraning rate, batch_size, and epochs
-   run ‘CNN-LSTM.py’
