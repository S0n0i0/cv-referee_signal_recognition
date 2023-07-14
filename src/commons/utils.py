# Functions
from pathlib import Path as Pt #Avoid confusion with class path
import os
#----------------Path----------------
class PATH:
    """ (static) Class holding different projects paths, used with .format(args) """
    CWD: str = str(Pt(__file__).parent.parent.parent) #Project Root
    SRC: str = os.path.join(CWD, "src")
    DATA: str = os.path.join(CWD, "data")#Path to data folder
    SAMPLES: str = os.path.join(CWD, "samples")
    MODELS: str = os.path.join(CWD, "models", "model_files", "model_{0}.p")
    DATA_FILES: str = os.path.join(CWD, "models", "data_files", "data_{0}.pickle")
    LOGS: str = os.path.join(Pt(__file__).parent.parent.parent, "logs")

