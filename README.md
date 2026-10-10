# A Sensor Trip Predictor Using Simulated Data

This is a machine learning model that predicts whether a machine is about to trip by looking at the last 30 readings from two sensors. 

The notebook includes the full machine learning workflow.

## How to run it

1. Set up the virtual environment and install libraries:
   python3 -m venv .venv
   source .venv/bin/activate
   pip install numpy pandas scikit-learn matplotlib ipykernel
2. Open `main.ipynb`, select the `.venv` kernel, and run all cells.

### Simulated data

Made a data generator to imitate sensor behaviour where each example is a window of 30 time steps from two sensors. Sensor A is around 1.0 and sensor B is around 5.0 arbitrarily. Both sensors have random noise and drift. Around 10% of windows are followed by a trip and around 80% of them have a warning, with the rest looking completely normal. The warnings are Sensor A gets noisier and sensor B slowly rises near end. These have different strengths and starting times to add randomness and subtleties.

### Workflow

Before a trip, the readings sometimes start behaving differently than usual. The model looks at a window of readings and gives a probability that a trip is coming. Different thresholds control how "suspicious" the model is.

1. Split the data into a training and test set keeping same fraction of trips in each.
2. Made a baseline to compare against.
3. Built a pipeline including a standard scaler and logistic regression.
4. Used 5-fold cross-validation on the training set to tune the hyperparameter C.
5. Retrained the model with the best C on all of the training data.
6. Chose the decision threshold using cross-validated predictions on the training data by looking at metrics.
7. Evaluated on the test set.
8. Wrote a function called `predict_trip`, which takes 2 windows of readings and a threshold, then returns a probability and if there will be a trip.

### Results
Caught 40 out of 53 trips
3 false alarms out of 547 normal windows
Precision: 0.93
Recall: 0.75
F1: 0.83
ROC AUC: 0.89
Average precision: 0.83


