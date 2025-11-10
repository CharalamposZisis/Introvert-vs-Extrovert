import logging
import os
from datetime import datetime 

LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log" # The inside of logs folder file (.log) structure 
log_path = os.path.join(os.getcwd(),"logs", LOG_FILE) # create logs directory path
os.makedirs(log_path, exist_ok=True) # create the logs directory if not exists (exist_ok = True)

LOG_FILE_PATH = os.path.join(log_path, LOG_FILE) # where the logs are going

logging.basicConfig(
    filename= LOG_FILE_PATH,
    format="[%(asctime)s] %(name)s %(lineno)d %(levelname)s - %(message)s", # format of loggings
    level = logging.INFO) # we use this to put it inside the file

if __name__ == "__main__":
    logging.info("Logging has started")