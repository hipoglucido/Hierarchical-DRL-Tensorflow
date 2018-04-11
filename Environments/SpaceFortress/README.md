# Setup
This project mainly uses the following preexisting software:
* [OpenAI Gym](https://gym.openai.com/)
* [Simple DQN](https://github.com/tambetm/simple_dqn)
* SpaceFortress

### OpenAI Gym
A toolkit for developing and comparing reinforcement learning algorithms. It supports teaching agents everything from walking to playing games like Pong or Go. Gym will create an environment in which the learning algorithm wiil learn the game.

### Simple DQN
Deep Q-learning agent for replicating DeepMind's results in paper "Human-level control through deep reinforcement learning". It is designed to be simple, fast and easy to extend. This algorithm will be used.to learn the game.

### Space Fortress
An old DOS game that will be learned by the network.

# Download
Download the repository. Make sure you execute the command in which you want to install the repository. Git creates a new folder called SpaceFortress where the repository will be placed.
```sh
sudo git clone https://github.com/Noswis/SpaceFortress.git
cd SpaceFortress
```
- - - -
# Installation
During the installation process, please make sure the terminal resides in the SpaceFortress folder.
### Neon
[Neon](https://github.com/NervanaSystems/neon) is Nervana's Python based Deep Learning framework and achieves the fastest performance on modern deep neural networks such as AlexNet, VGG and GoogLeNet. Designed for ease-of-use and extensibility. Simple DQN  was built on neon, so we need to install this.Neon free software which can be cloned from their github page.

Install prerequisites:
```sh
sudo apt-get install libhdf5-dev libyaml-dev libopencv-dev pkg-config
sudo apt-get install python python-dev python-pip python-virtualenv
sudo apt-get install python-opencv
```

Check out and compile the code:

```sh
git clone https://github.com/NervanaSystems/neon.git
cd neon
sudo make sysinstall
```
Clean up the dowloaded folder
```sh
cd ..
sudo rm -rf neon
```
### Gym
Install prerequisites:
```sh
# Use 'sudo apt-get install pip' to get pip if you haven't already
sudo pip install -r gym/requirements.txt
```
Connect python to gym
```sh
# add the gym directory to the python path in bashrc
echo "export PYTHONPATH=$PYTHONPATH:$PWD" >> ~/.bashrc
source ~/.bashrc
```
Note: please make sure the path to the SpaceFortress folder does not have any spaces

Check if it is installed correctly:
```sh
python
import gym
```

If a new line appears without anything of importance that happened, gym is installed correctly.

Exit the python environment using the ' exit() ' command.
### Cairo
Cairo is a 2D graphics library with support for multiple output devices. Currently supported output targets include the X Window System (via both Xlib and XCB), Quartz, Win32, image buffers, PostScript, PDF, and SVG file output. Experimental backends include OpenGL, BeOS, OS/2, and DirectFB. The game uses this engine to render scenes. We can get is very easily with aptitude package manager:

```sh
sudo apt-get install libcairo2-dev
```
If you run into any problems installing, please visit [this page](https://www.cairographics.org/download/)
### Shared Libraries
Go to the Game folder
The learning environment needs shared libraries, which can be built from the sourcecode of the SpaceFortress game. In the folder 'game' are multiple subfolders which are subtasks of the original game.
* SF is the game with most elements included
* AIM is the stripped version of the game which focusses on the aiming task
* SFC is the stripped version of the game which focusses on the control task

Make sure you have the clang compiler installed
```sh
sudo apt-get install clang
```

The Makefile has several build options, as listed below:
* SF
* AIM
* SFC
* all
* clean

For the installation, the only command of importance is `make all`. Follow the next steps to create the shared libraries from the game's sourcecode:
```sh
cd Game
sudo make all    # makes all versions of the game
```


### Other dependencies
A list with dependencies we encountered during installation/runtime ourselves.
```sh
sudo apt-get install python-xlib
sudo pip install pynput
sudo pip install pathlib
sudo pip install matplotlib
sudo apt-get install gtk2.0
sudo apt-get install libgtk2.0-dev
sudo apt-get install libgtk-3-dev
sudo apt-get install pkg-config
```

### Testing the environment
In the main installation directory, there is a symbolic link called space_fortress. This link reffers to the directory gym/env/space_fortress and was added for convieniency. In this folder execute run.py to test if the environment is working, using the following command:
```sh
cd space_fortress
python run.py
```
If a graphical window appears and the game is running, everything has been installed correctly.
You can skip to the part below and go to the next section *Usage*.

If an error occurs as below, please head over to the *Troubleshoot* section
```sh
"OpenCV Error: Unspecified error (The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Carbon support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script) in cvNamedWindow, file /io/opencv/modules/highui/src/window.cpp, line 565"
```
- - - -
# Usage
### Running the game

Inside the space_fortress map, run:
```sh
python run.py
```
Then execute the scripts with the keys: "z, x, c, v, b, n, m"
In the constants file, all settings for running the game can be found and tweaked.

### Start Training
The network can be trained with the shell script train.sh located in the Simple_DQN folder. This script calls
the main python script in src along with parameters which specify where to save the weights and results. For instance, the command below will save the training data in ``runs/SFC/MyFirstTraining``.
```sh
./train.sh SFC MyFirstTraining cpu
```
Instead of SFC, it is also possible to train AIM, SFS and SF.

### Resume Training
With the resume script, a halted training process can be resumed.
```sh
./resume.sh SFC MyFirstTraining cpu
```
The learning algorithm uses 'epochs', which can be seen as checkpoints. The epochs are saved as .prm files in the weights folder, in the above example, the path to the weights would be ``runs/SFC/MyFirstTraining/weights``. The script will automatically find the most recent checkpoint and load the data. The results and statistics of the training are stored in .csv files. Each time the training is resumed, a new .csv file is created witch a session ID which makes it easier to identify training sessions.

Note: Please make sure the network trained until one epoch before trying to resume. The script will otherwise not work, because no checkpoint was saved.


### Plot training
The training can be plotted with plot.sh. For instance:
```sh
./plot.sh SFC MyFirstTraining outputFile
```
The last parameter is optional, this is the name of the .png file where the
graphs will be stored. When removing the last parameter, the graphs are plotted
in a newly created window.

### Get best epoch
The best epoch of a training can be retrieved by running get_best.sh. For instance:
```sh
./get_best.sh SFC MyFirstTraining
```
This will save the best epoch in the snapshots folder

### Play best epoch
To play the best epoch of a training, run ./play_best.sh. It can be run in this way:
```sh
./play_best.sh SFC MyFirstTraining
```
It is not needed to first run get_best before play_best, the play_best automatically
checks if the .prm file is located in the snapshots folder. If not, it runs get_best
first.
- - - -
# Troubleshoot
When setting up the repository, you might encounter bugs. This section covers the bugs our team found when setting up our repository on multiple systems.
### OpenCV Bug
Retry to install the correct openCV dependencies
```sh
sudo apt-get install libopencv-dev python-opencv
```

Open a new terminal and execute the commands:
```sh
python
import cv2
cv2.__file__
```

The output line should look like this:
``/usr/lib/python2.7/dist-packages/cv2.x86_64-linux-gnu.so``

If cv2 is imported from another location, browse to that location and delete the directory and/or files.
Keep repeating the above steps until cv2 is imported from the file cv2.x86_64-linux-gnu.so, as shown above.
Afterwards, exit the python environment rerun the following command:
```sh
python run.py
```
<!-- If everything was done correctly, the game should be working by now. -->

### Resuming
If you get something like this when resuming a training, check if the values of the constants.py are the same as when you started training. This indicates that the
settings are not the same as the settings you started training with.
```sh
TypeError: ary.size 2560 != self.size 2048
```
