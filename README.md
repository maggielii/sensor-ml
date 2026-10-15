# A Sensor Trip Predictor Using Simulated Data Applied as a Pump Leak Detector on Real Data

This is a machine learning model that predicts whether a machine is about to trip by looking at the last 30 readings from two sensors. 

After building the pipeline on simulated data, I wanted to see if the same workflow can be applied to real sensor data. This is demonstrated in `hydraulic_rig.ipynb`.

The notebooks include the full machine learning workflows.

1. Set up the virtual environment and install libraries:
```
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn matplotlib ipykernel
```
2. For Part 1: Open `main.ipynb`, select the `.venv` kernel, and run all cells.
3. For Part 2: Download the dataset from https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems and unzip it. Put the files directly in `data/hydraulic/` ex. `data/hydraulic/profile.txt`. Open `hydraulic_rig.ipynb`, select the `.venv` kernel, and run all cells.

# Part 1:

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

# Part 2: Detecting pump leaks

### Data
- UCI "Condition Monitoring of Hydraulic Systems" recorded on a lab test rig. Note: The data is not included in this repo due to large files. 
- The rig repeats a 60-second cycle 2205 times while the cooler, valve, pump, and accumulator are deliberately worn to different levels
- 17 sensors of pressure, flow, temperature, vibration, motor power, and a few calculated values

### Workflow
- Summarized each sensor's readings in each cycle as mean, std and slope like the window summaries in part 1
- Same workflow as part 1 with train/test split, baseline, scaler + logistic regression, tuned C from 5-fold cross-validation, chosen threshold from cross-validated predictions on the training data, and test set evaluated

### Results
Baseline of always predicting "no leak": 55% accuracy

Caught 197 out of 197 leaks

0 false alarms

Precision: 1.0
Recall: 1.0
F1: 1.0
ROC AUC: 1.0

### Future Improvement 
- This is a lab rig with a constant cycle and deliberately worn parts, in the future I aim to explore faults that are not controlled

#### Data Citation
Helwig, N., Pignanelli, E., & Schtze, A. (2015). Condition monitoring of hydraulic systems [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CW21.