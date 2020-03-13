# coding : utf-8

import logging
import os

def init_logging(log_file=None, file_mode='w', overwrite_flag=False, log_level=logging.DEBUG):
    # basically, the basic log offers console output
    consoleHandler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s[%(levelname)s]: %(message)s')
    consoleHandler.setFormatter(formatter)
    
    logging.getLogger().setLevel(log_level)
    logging.getLogger().addHandler(consoleHandler)
    
    
    if not log_file or log_file == '':
        print '----------------------------------------------------------'
        print 'No log file is specified. The log information is only displayed in command line.'
        print '----------------------------------------------------------'
        return
    
    # check that the log_file is already existed or not
    if not os.path.exists(log_file):
        locationDir = os.path.dirname(log_file)
        if not os.path.exists(locationDir):
            os.makedirs(locationDir)
        
        fileHandler = logging.FileHandler(filename=log_file, mode=file_mode)
        fileHandler.setFormatter(formatter)
        logging.getLogger().addHandler(fileHandler)

        print 'The logging is successfully init. The log file is created.'
    else:
        if overwrite_flag:
            print 'The file [%s] is existed. And it is to be handled according to the arg [file_mode](the default is \'w\').' % log_file
            fileHandler = logging.FileHandler(filename=log_file, mode=file_mode)
            fileHandler.setFormatter(formatter)
            logging.getLogger().addHandler(fileHandler)
        else:
            print 'The file [%s] is existed. The [overwrite_flag] is False, please change the [log_file].' % log_file
            quit()
            

def test():
    log_file='./test.log'
    file_mode='w'
    init_logging(log_file=log_file, file_mode=file_mode, overwrite_flag=True, log_level=logging.DEBUG)
        
        
if __name__=='__main__':
    test()
    logging.info('test info')
        
