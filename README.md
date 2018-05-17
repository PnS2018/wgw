# WaldoGoneWild

## Project Description
In our project we try to apply Deep Learning techniques with Computer Vision tools to automatically find Waldo in any "Where is Waldo?" picture.

## How to run this project locally
First install all the packages in the requirements.txt file on your local machine or venv.
After that you have to add the project folder to your Source Root. In PyCharm: Left Click on Folder -> Mark Directory as -> Source Root.

All global configurations are handled in the `config.yml` file found at the root.
Our data set can be found in the `data` folder, where we have separate folders for testing and training data.
Our models are contained in the ModelManager found in the `model` directory.

#### Training
To train the model, specify a version number in the config file and run the script `train_model.py`. This will save the model to an .h5 file.

#### Validating
To validate our model run the script `check_precision.py`, which will give you the number of test samples checked and the number of false positives and false negatives.

#### Running
To run our project use the script `run.py` which will take an image as input and mark the cell in which it thinks Waldo can be found.

## Who to speak to?
- Michel
- Gina
- Eric