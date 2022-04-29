# Python Multithreading Hypervisor for acccurate estimation of function process time

from processes import Processes

import threading
import logging
import os

logging.basicConfig(format='[%(threadName)s]: %(message)s', level=logging.DEBUG,)

  
if __name__ == "__main__":

    inputList = ['demoSet/set3overlap/1.png', 'demoSet/set3overlap/2.png']
    
    # Lock object for synced access
    threadLock = threading.Lock()

    # Thread init
    thread1 = threading.Thread(target=Processes.interpolationDemo, args=(threadLock, inputList))
    thread1.setDaemon(True)

    # Main Thread Descriptor
    mainThread = threading.current_thread()

    # Thread1 Start
    thread1.start()

    # Wait for thread to complete the process
    thread1.join()
    print('done')


    '''
    Thread List Iteration for future use
    '''
    # # Thread List Iteration
    # for thread in threading.enumerate():
    #     if thread is mainThread:
    #         continue

    #     # Wait for thread to complete the process
    #     thread.join()
    #     logging.debug('Joined %s', thread.getName())

    #     # Thread daemon done.
    #     if thread.is_alive() != False:
    #         logging.debug('Thread %s timeout', thread.getName())
    #     logging.debug('Thread %s done', thread.getName())