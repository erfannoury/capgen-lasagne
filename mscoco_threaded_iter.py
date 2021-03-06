from __future__ import print_function, division
try:
    import Queue
except ImportError:
    import queue as Queue
import threading
import time
import numpy as np
import warnings
import os
import skimage
from skimage.transform import resize
from skimage.io import imread
import itertools
from itertools import izip_longest
import time
from time import sleep
from datetime import datetime
import cPickle as pickle
from pycocotools.coco import COCO
from nltk.tokenize import TreebankWordTokenizer

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

class COCOThread(threading.Thread):
    """
    The thread that has two queues, the "queue"
    will contain the objects for which we will
    call the data augmentation function and then
    put the result in the "out_queue"
    """
    def __init__(self, queue, out_queue, prep_func):
        threading.Thread.__init__(self)
        self.queue = queue
        self.out_queue = out_queue
        self.prep_func = prep_func

    def run(self):
        while True:
            annIds, seqlen = self.queue.get()
            try:
                minibatch = self.prep_func(annIds, seqlen)
                self.out_queue.put(minibatch)
                self.queue.task_done()
            except IOError:
                pass

class COCOCaptionDataset():
    """
    The main iterator for the COCO Caption dataset.
    Each item of the iterator is a batch that is
    ready to be uploaded to GPU. It also has a
    separate thread to prepare the next batch
    concurrently on CPU.

    Parameters
    ----------
    images_path: str
        path to the folder that contains the images corresponding
        to the annotations set
    annotations_path: std
        path to the annotation file that will be used
    buckets: dict
        a dictionary of sequence lengths and annotation Ids that
        its captions length is less than that sequence length
        [sl_0] -> [annId_1, ..., annId_m]
    bucket_minibatch_sizes: dict
        a dictionary of sequence lengths and minibatch sizes for
        that minibatches of that sequence lengths
    word2idx: dict
        a dictionary converting each word to the corresponding index
        some special cases:
            word2idx['<PAD>'] = 0
            word2idx['<GO>'] = 1
            word2idx['<EOS>'] = 2
    mean_im: np.ndarray
        a 4-tensor of size (1, 3, 224, 224) of values of the mean image
        for the CNN model that'll be used
    shuffle: bool (True)
        a boolean indicating whether the minibatches should be shuffled or not
    """
    def __init__(self, images_path, annotations_path, buckets, bucket_minibatch_sizes, word2idx, mean_im, shuffle=True):
        self.buckets = buckets
        self.word2idx = word2idx
        self.bucket_minibatch_sizes = bucket_minibatch_sizes
        self.buffer_size = 16
        self.input_qsize = 64
        self.min_input_qsize = 16
        self.total_max = 0
        self.mean_im = mean_im
        self.tokenizer = TreebankWordTokenizer()
        self.annotations_path = annotations_path
        self.images_path = images_path
        self.shuffle = shuffle
        self._initialize()
        self.queue = Queue.Queue()
        self.out_queue = Queue.Queue(maxsize=self.buffer_size)
        self._init_queues()

    def _initialize(self):
        for sl in self.buckets.keys():
            self.total_max += int(np.ceil(len(self.buckets[sl]) / self.bucket_minibatch_sizes[sl]))
        self.coco = COCO(self.annotations_path)
        self.reset()

    def __prep_minibatch__(self, annIds, seqlen):
        """
        Parameters
        ----------
        annIds: list
            list of integers representing the annotation ids
        seqlen: int
            maximum length of sequence for this minibatch

        Returns
        -------
        im_minibatch: np.ndarray
            a tensor of size `minibatch size, 3, 244, 244`
        caption_in_minibatch: np.ndarray
            a matrix of size `minibatch size, seqlen`
        caption_out_minibatch: np.ndarray
            a matrix of size `minibatch size, seqlen`
        """
        annIds = filter(None, annIds)
        minibatch_size = len(annIds)
        anns = []
        for id in annIds:
            anns.append(self.coco.loadAnns(id)[0])
        im_minibatch = np.zeros((minibatch_size, 3, 224, 224), dtype=np.float32)
        caption_in_minibatch = np.zeros((minibatch_size, seqlen), dtype=np.int32)
        caption_in_minibatch[:, 0] = self.word2idx['<GO>']
        caption_out_minibatch = np.zeros((minibatch_size, seqlen), dtype=np.int32)
        for i, ann in enumerate(anns):
            im_file_name = os.path.join(self.images_path, self.coco.loadImgs(ann['image_id'])[0]['file_name'])
            try:
                im = skimage.io.imread(im_file_name).astype(np.float32)
                if im.ndim == 2:
                    im = np.tile(im[:, :, np.newaxis], (1, 1, 3))
                im = skimage.transform.resize(im, (224, 224), preserve_range=True).transpose((2, 0, 1)) # c01
            except ValueError:
                im = np.zeros((3, 224, 224), dtype=np.float32)
                print("Error in reading image {}".format(im_file_name))
            im_minibatch[i, ...] = im[::-1, :, :]
            caption = ann['caption']
            tokens = self.tokenizer.tokenize(caption.lower())
            # caption input: <GO>, token_1, ..., token_m, <PAD>, ...
            # caption output: token_1, ..., token_m, <EOS>, <PAD>, ...
            for j, t in enumerate(tokens):
                caption_in_minibatch[i, j+1] = self.word2idx[t]
                caption_out_minibatch[i, j] = self.word2idx[t]
            caption_out_minibatch[i, len(tokens)] = self.word2idx['<EOS>']
        im_minibatch -= self.mean_im
        return (im_minibatch, caption_in_minibatch, caption_out_minibatch)

    def _init_queues(self):
        self.th = COCOThread(self.queue, self.out_queue, self.__prep_minibatch__)
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
        self.max_size = 0
        self.seqlens = (i for i in sorted(self.buckets.keys()))
        self.proceed_bucket()

    def proceed_bucket(self):
        try:
            self.current_seqlen = self.seqlens.next()
        except StopIteration:
            self.reset()
        if self.shuffle:
            np.random.shuffle(self.buckets[self.current_seqlen])
        self.max_size += int(np.ceil(len(self.buckets[self.current_seqlen]) / self.bucket_minibatch_sizes[self.current_seqlen]))
        self.current_bucket = grouper(self.buckets[self.current_seqlen], self.bucket_minibatch_sizes[self.current_seqlen])

    def _step(self):
        if self.n_consumed >= self.total_max:
            self.reset()
            raise StopIteration("End of epoch")
        if self.n_consumed >= self.max_size:
            self.proceed_bucket()
        if self.queue.qsize() <= self.min_input_qsize:
            for _ in xrange(self.input_qsize):
                try:
                    self.queue.put((self.current_bucket.next(), self.current_seqlen))
                except StopIteration:
                    pass
        minibatch = self.out_queue.get()
        self.n_consumed += 1
        return minibatch

if __name__ == '__main__':
    resnet_weights = pickle.load(open('/home/noury/modelzoo/resnet50.pkl', 'rb'))
    mean_im = resnet_weights['mean_image'].reshape((1, 3, 224, 224)).astype(np.float32)
    images_path = '/home/noury/datasets/mscoco/train2014/'
    annotations_file_path = '/home/noury/datasets/mscoco/annotations/captions_train2014.json'
    coco_captions = pickle.load(open('coco_captions_trainval2014.pkl', 'rb'))
    train_buckets = coco_captions['train buckets']
    bucket_minibatch_sizes = {16:64, 32:32, 64:16}
    wordset = coco_captions['raw wordset']
    word2idx = {}
    word2idx['<PAD>'] = 0
    word2idx['<GO>'] = 1
    word2idx['<EOS>'] = 2
    for i, w in enumerate(wordset):
        word2idx[w] = i+3
    coco = COCOCaptionDataset(images_path, annotations_file_path, train_buckets, bucket_minibatch_sizes, word2idx, mean_im, True)
    for e in xrange(2):
        print("Starting epoch", e+1)
        i = 0
	now = datetime.now()
        for im, capin, capout in coco:
            i += 1
            print(datetime.now() - now)
            print(i, '\t', im.shape, capin.shape, capout.shape)
            now = datetime.now()
