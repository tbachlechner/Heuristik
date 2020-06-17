""" Initialization of heuristik package. Run this first """

from config import *
from config import config
# Config prompts for APS keys and stores them on disk, and sets environment variables
config()
from data_manager import *
from training import *
from call import *