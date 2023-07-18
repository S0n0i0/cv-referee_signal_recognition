# Classes definitions, constants, ...

from pathlib import Path as Pt #Avoid confusion with class path
import os
from enum import Enum
import pickle
import tensorflow as tf
from positional_encodings.tf_encodings import TFPositionalEncoding1D, TFSummer
#from tensorflow.keras.layers import Layer

class PATH:
    """ (static) Class holding different projects paths, used with .format(args) """
    CWD: str = str(Pt(__file__).parent.parent.parent) #Project Root
    SRC: str = os.path.join(CWD, "src")
    DATA: str = os.path.join(CWD, "data")#Path to data folder
    SAMPLES: str = os.path.join(CWD, "samples")
    MODELS: str = os.path.join(CWD, "models", "model_files", "model_{0}.{1}")
    DATA_FILES: str = os.path.join(CWD, "models", "data_files", "data_{0}.pickle")
    LOGS: str = os.path.join(Pt(__file__).parent.parent.parent, "logs")

class model_type(Enum):
    NUMBERS = 0
    PENALTY = 1
    FOULS = 2

class portable_model:
    type: model_type
    model: any

    def __init__(self, type: model_type, calibration: bool = False) -> None:
        match type:
            case model_type.NUMBERS:
                self.type = type
                if(calibration):
                    self.path = PATH.MODELS.format("number_calibrated","p")
                    self.model = pickle.load(open(self.path, 'rb'))
                else:
                    self.path = PATH.MODELS.format("number","p")
                    self.model = pickle.load(open(self.path, 'rb'))
            case model_type.PENALTY:
                self.type = type
                if(calibration):
                    self.path = PATH.MODELS.format("penalty_calibrated","p")
                    self.model = pickle.load(open(self.path, 'rb'))
                else:
                    self.path = PATH.MODELS.format("penalty","p")
                    self.model = pickle.load(open(self.path, 'rb'))
            case model_type.FOULS:
                self.type = type
                self.path = PATH.MODELS.format("fouls","keras")
                self.model = tf.keras.models.load_model(self.path)
            case _:
                raise TypeError("Class does not exists")

'''class PositionalEncoding(Layer):
    def __init__(self, size, **kwargs):
        super().__init__(**kwargs)
        self.size = size

    def get_config(self):
        config = super().get_config()
        config.update({
            "size": self.size
        })
        return config

    def call(self, inputs):
        p_enc_2d = TFPositionalEncoding1D(self.size)
        add_p_enc_2d = TFSummer(TFPositionalEncoding1D(self.size))
        return add_p_enc_2d(inputs) - p_enc_2d(inputs)'''