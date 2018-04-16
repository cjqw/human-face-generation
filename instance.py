from config import get_config

import os

class LFWInstance(object):
    def __init__(self):
        data_path = get_config("data-path") or "./"
