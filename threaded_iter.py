from __future__ import print_function
try:
    import Queue
except ImportError:
    import queue as Queue
import threading
import time
import numpy as np
import warnings
import os
import itertools
import time
from time import sleep

def data_augmentation(img_addr):
    """
    This is the time-consuming function that may contain
    IO operations, and will be processed in the CPU.
    This is the function that can be parallelized the GPU
    operations. While the GPU processes the current batch of
    data, this function in a separate thread will prepare the
    next batch.
    """
    sleep(1 + np.random.random())
    print("The request for image {} is served.".format(img_addr))
    return 'augmented image {}'.format(img_addr)


class TheThread(threading.Thread):
    """
    The thread that has two queues, the "queue" 
    will contain the objects for which we will
    call the data augmentation function and then 
    put the result in the "out_queue"
    """
    def __init__(self, queue, out_queue):
        threading.Thread.__init__(self)
        self.queue = queue
        self.out_queue = out_queue

    def run(self):
        while True:
            addr = self.queue.get()
            try:
                res = data_augmentation(addr)
                self.out_queue.put(res)
                self.queue.task_done()
            except IOError:
                pass

class TheDataset():
    """
    The main iterator for the dataset.
    Each item of the iterator is a batch that is 
    ready to be uploaded to GPU. It also has a 
    separate thread to prepare the next batch
    concurrently on CPU.
    """
    def __init__(self, mb_size):
        self.mb_size = mb_size
        self.buffer_size = 5
        self.input_qsize = 10
        self.min_input_qsize = 5
        self.n_consumed = 0
        self.max_value = 30
        self.image_addresses = (i for i in xrange(self.max_value)) # a generator
        self.queue = Queue.Queue()
        self.out_queue = Queue.Queue(maxsize=self.buffer_size)
        self._init_queues()

    def _init_queues(self):
        self.th = TheThread(self.queue, self.out_queue)
        self.th.setDaemon(True)
        self.th.start()
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
    
    def next(self):
        return self._step()
    
    def reset(self):
        self.n_consumed = 0

    def _step(self):
        print("\tstatus -> n_consumed={}, qsize={}".format(self.n_consumed, self.queue.qsize()))
        if self.n_consumed >= self.max_value:
            self.reset()
            raise StopIteration("End of epoch")
        if self.queue.qsize() <= self.min_input_qsize:
            print('emergency filling of the queue')
            for x in xrange(self.input_qsize):
                try:
                    self.queue.put(self.image_addresses.next())
                except StopIteration:
                    pass
        aug_img = self.out_queue.get()
        self.n_consumed += 1
        return aug_img

if __name__ == '__main__':
    td = TheDataset(3)
    while True:
        print('using', td.next())
        sleep(np.random.random() * 3 + 1)
