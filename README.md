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
The software will work in 3 different sections with respect to what we will explain in the following section

## While in front of the scorers table

Assumptions:
* The referee is in front of the table
* Referee knows the referee code

Approach:
* Single task approach:
> * Number recognition
> * Type of foul
> * Type of penality
* Task combination

### Number recognition

Assumption:
* Numbers up to 99 (Trentino)

Hand casuistry:
* Single number:
> * 0-5: one hand on the palm side
> * 6-10: two hand numbers on the palm side
* Double digit:
> * 11-15: fist plus hand on the palm side
> * 16-99: first number with the back of the hand and the second one with the palm side

### Type of foul

### Type of penalty

### Task combination

## During the whole game

* Referee typically has different colour palette on clothes w.r.t. the teams that are playing.
> * Inizialmente mettiamo il codice colore predominante noi. Poi riconosci colori predominanti nel campo.

* Visualizziamo tutto il campo di gioco, cercando di evitare di inquadrare gli spalti.
> * In seguito puntiamo su una migliore angolazione della telecamera.

