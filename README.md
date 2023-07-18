# Referee signal recognition
## How to setup and use the project

### Installation
1. Firstly, you need clone the repository
    ```sh
    git clone https://github.com/S0n0i0/cv-referee_signal_recognition
    ```
2. In the folder where cv-referee_signal_recognition is located (in ../cv-referee_signal_recognition) execute:
    ```sh
    pip install -e cv-referee_signal_recognition
    ```
3. Install requirements (in cv-referee_signal_recognition)
    ```sh
    pip install -r requirements.txt
    ```

### Usage
To start using our solution, you need to go into the src directory and search for table_report directory.
After entering in the directory, you need to start the program with
```sh
python table_report.py
```

If you want to execute the various components (namely `hands_gesture_recognition` and `fouls_gesture_recognition`) singularly you have to go into their directory and execute the file with the same name of the directory as specified before.

Other programs that you may want to run are:
* `hands_gesture_recognition`:
* `fouls_gesture_recognition`:
  * `collect_input.py`: analyze samples in the homonymous directory and extract features for the training part
  * `train_model.py`: train the neural network with data presents in the directory `data/fouls` and create a trained model into the directory `models/model_files`

## While in front of the scorers table

Assumptions:
* The referee is in front of the table
* Referee knows the referee code

Approach:
* Single task approach:
  * Number recognition
  * Type of foul
  * Type of penality
* Task combination

### Number recognition

Assumption:
* Numbers up to 99 (Trentino)

Hand casuistry:
* Single number:
  * 0-5: one hand on the palm side
  * 6-10: two hand numbers on the palm side
* Double digit:
  * 11-15: fist plus hand on the palm side
  * 16-99: first number with the back of the hand and the second one with the palm side

### Type of foul

Labels:
\begin{myitemize}
* Block during an attack action
* Breakthrough
* Block during a defense action
* Excessive use of elbows
* Hand checking
* Head hit
* Hit while throwing the ball
* Holding
* Hooking
* Illegal use of hands
* Push

### Type of penalty

### Task combination

## During the whole game

* Referee typically has different colour palette on clothes w.r.t. the teams that are playing.
  * Initially we'll fixed color code
  * Then we'll use one of these approaches:
    * Recognize the predominant color in the game field (players) and then take the person with different color
    * Do a calibration before the game, in order to pick the predominant color of the referee
* Visualize all the game field, trying to avoid the stands (avoiding to framing them or cropping the image)

