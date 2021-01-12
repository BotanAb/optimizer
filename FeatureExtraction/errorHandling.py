import sys
import traceback
import logging
from datetime import datetime

datetime = datetime.now()
time = datetime.strftime("%H:%M:%S")
LOG_FILENAME = "./tmp/crash_report.out"
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)

logging.debug("Dit is een crash report op " + time)

def generic_exception_handler():
    print("------------------------")
    print("Programma heeft een exception")
    traceback.print_exception(*sys.exc_info())
    logging.exception('er is een exception geworpen, kijk hieronder voor de traceback')
    print("------------------------")
