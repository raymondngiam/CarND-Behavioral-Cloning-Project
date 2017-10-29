# Udacity Self Driving Car Nanodegree
## Behavioral Cloning

### Overview
This is a project for Udacity's Self Driving Car Nanodegree. The objective is to train a convolutional neural network for autonomous prediction of steering angles from camera images.

### Dependencies
- python==3.5.2
- numpy
- matplotlib
- opencv3
- h5py
- scikit-learn
- pandas
- flask-socketio
- ffmpeg
- imageio=2.1.2
- tensorflow==0.12.1
- keras==1.2.1

### Installation
**Install Anaconda:**

Follow the instructions on the [Anaconda download site](https://www.continuum.io/downloads).

**Create environment:**

For users with CPU only, running this command will create a new `conda` environment that is provisioned with all libraries you need to run the iPython notebooks.

```
$ conda env create -f environment.yml
```

For users with GPU, ensure you have installed CUDA toolkit 8.0 and CuDNN v5. Then run the following command:

```
$ conda env create -f environment-gpu.yml
```

**Uninstall environment:**

To uninstall the environment:

```
$ conda env remove -n behavioral-cloning
```

**Activate environment:**

In order to use the environment, you will need to activate it. This must be done **each** time you open a new terminal window. 

```
$ source activate behavioral-cloning
```

To exit the environment, simply close the terminal window or run the following command:

```
$ source deactivate behavioral-cloning
```

### How to run

Download the [Udacity simulator](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip) for Linux. 

Record the training data in the simulator and store it in the directory `/data`. This directory should consist of a `/IMG` image directory and a `driving_log.csv` file.

To create the model, run the following command to train the model with generator batch size of 32, dropout rate of 0.5 for 5 epochs:

```
$ python model.py --epochs 5 --batch_size 32 --dropout_rate 0.5
```

Feel free to try any other combinations of the hyperparameters. 

The default outputs of this script are saved as `model.h5` (a Keras model file) and `model.p` (a binary file storing the training log). The name of the output files can be changed with the command line flag `--output_name`.

For more details of the command line flags, please refer to `model.py`.

To run the trained model in the Udacity simulator, first launch the simulator and select "AUTONOMOUS MODE". Then run:

```
$ python drive.py model.h5
```

To create a first person video of the trained model, run the following command to save all images seen by the agent into image directory `run1`.

```
$ python drive.py model.h5 run1
```

Then run the next command to create a video based on images found in the directory.

```
$ python video.py run1
```

The name of the video will be the name of the directory followed by '.mp4', so, in this case the video will be `run1.mp4`.

### Video demo

You can see the video demo of the final trained model at [here](https://vimeo.com/240369937).

### Final report

You can view the final report of this project at [writeup.pdf](https://github.com/raymondngiam/CarND-Behavioral-Cloning-Project/blob/master/writeup.pdf).
