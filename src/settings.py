# -*- coding: utf-8 -*-
import os
import utils

#Session timestamp
SESSION_TS = utils.get_timestamp()

#Running settigns

#Monitoring settings
USE_TB = 0


#Dir settings

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
DATA_DIR = os.path.join(ROOT_DIR, 'data')




#Auxiliary
logger = utils.Logger()
if USE_TB:
      TB_DIR = os.path.join(DATA_DIR, 'tensorboard', SESSION_TS)
      os.makedirs(TB_DIR)
      logger.info("TB folder created " + TB_DIR)


