from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from src.commons.utils import PATH
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

from pathlib import Path as Pt #Avoid confusion with class path

category = "fouls"
dataset_size = 30

PATH.SAMPLES = os.path.join(Pt(__file__).parent.parent.parent, "samples")
PATH.DATA = os.path.join(Pt(__file__).parent.parent.parent, "data")
PATH.LOGS = os.path.join(Pt(__file__).parent.parent.parent, "logs")
PATH.MODELS = os.path.join(Pt(__file__).parent.parent.parent, "models")
model_path = os.path.join(PATH.MODELS,"model_files")

category_path = os.path.join(PATH.DATA, category)

actions = np.array(os.listdir(category_path))

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
print(actions)
for action in actions:
    action_path = os.path.join(category_path, action)
    # number of files contained into action
    for sequence in np.array(os.listdir(action_path)):
        window = []
        # number of svo files in the directory action_path
        sequences_files = os.listdir(os.path.join(action_path, sequence))
        for frame_num in range(min(len(sequences_files),dataset_size)):
            res = np.load(os.path.join(action_path, sequence, "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)


# _ Build and train LSTM Neural Network
tb_callback = TensorBoard(log_dir=PATH.LOGS)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,154)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback]) #epochs=2000

print(model.summary())

res = model.predict(X_test)
print("Prediction: ",actions[np.argmax(res[0])]," -> ",actions[np.argmax(y_test[0])])
#print("Example prediction: ",actions[np.argmax(res[2])]," = ",actions[np.argmax(y_test[2])])

yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
print("Multilabel confusion matrix: ",multilabel_confusion_matrix(ytrue, yhat))
print("Accuracy score:")
print(accuracy_score(ytrue, yhat))

model.save(os.path.join(model_path,'model_fouls.keras'))

del model