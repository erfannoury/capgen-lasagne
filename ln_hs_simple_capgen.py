from __future__ import division, print_function
import logging
import numpy as np
import scipy as sc
import skimage
from skimage import transform
import theano
import theano.tensor as T
import lasagne
import sys
import cPickle as pickle
import gc
from datetime import datetime
from collections import OrderedDict
from mscoco_threaded_iter import COCOCaptionDataset
sys.path.append('/home/noury/codevault/Recipes/modelzoo/')
sys.path.append('/home/noury/codevault/seq2seq-lasagne/')
from resnet50 import build_model
from CustomLSTMLayer import LNLSTMLayer
from HierarchicalSoftmax import HierarchicalSoftmaxLayer

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(message)s', '%m/%d/%Y %I:%M:%S %p')
    fh = logging.FileHandler('ln_hs_mscoco_captions.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info('Loading the ResNet50 model.')
    # First get the ResNet50 model
    resnet_weights_path = '/home/noury/modelzoo/resnet50.pkl'
    resnet = build_model()
    model_params = pickle.load(open(resnet_weights_path, 'rb'))
    lasagne.layers.set_all_param_values(resnet['prob'], model_params['values'])
    mean_im = model_params['mean_image'].reshape((1, 3, 224, 224)).astype(np.float32)

    # Load the files needed for the MS COCO Captions dataset
    train_images_path = '/home/noury/datasets/mscoco/train2014'
    valid_images_path = '/home/noury/datasets/mscoco/val2014'
    train_annotations_filepath = '/home/noury/datasets/mscoco/annotations/captions_train2014.json'
    valid_annotations_filepath = '/home/noury/datasets/mscoco/annotations/captions_val2014.json'
    coco_captions = pickle.load(open('coco_captions_trainval2014.pkl', 'rb'))
    train_buckets = coco_captions['train buckets']
    valid_buckets = coco_captions['valid buckets']
    wordset = coco_captions['raw wordset']
    word2idx = {}
    word2idx['<PAD>'] = 0
    word2idx['<GO>'] = 1
    word2idx['<EOS>'] = 2
    for i, w in enumerate(wordset):
        word2idx[w] = i+3
    idx2word = map(lambda x: x[0], sorted(word2idx.items(), key=lambda x: x[1]))

    bucket_minibatch_sizes = {16:64, 32:32, 64:16}


    logger.info('Creating global variables')
    CONTINUE = True
    HIDDEN_SIZE = 2048
    EMBEDDING_SIZE = 300
    WORD_SIZE = len(idx2word)
    DENSE_SIZE = 1024
    ORDER_VIOLATION_COEFF = 0.2
    RNN_GRAD_CLIP = 32
    TOTAL_GRAD_CLIP = 64
    TOTAL_MAX_NORM = 128
    RESNET_ADAM_LR = theano.shared(np.float32(0.00001 * 2), 'resnet adam lr')
    RECURR_ADAM_LR = theano.shared(np.float32(0.0001 * 2), 'recurrent adam lr')
    RESNET_SGDM_LR = theano.shared(np.float32(0.0001 * 2), 'resnet sgdm lr')
    RECURR_SGDM_LR = theano.shared(np.float32(0.001 * 2), 'recurrent sgdm lr')
    EPOCH_LR_COEFF = np.float32(0.5)
    NUM_EPOCHS = 15
    ADAM_EPOCHS = 3

    if CONTINUE:
        import glob
        param_values = glob.glob('ln_hs_param_values_*.pkl')
        max_epoch = max(map(lambda x: int(x[len('ln_hs_param_values_'):-len('.pkl')]), param_values))
        logger.info('Continue training from epoch {}'.format(max_epoch + 1))
        logger.info('Setting previous parameter values from epoch {}'.format(max_epoch))
        if max_epoch >= ADAM_EPOCHS:
            for _ in xrange(max_epoch - ADAM_EPOCHS):
                RESNET_SGDM_LR.set_value(RESNET_SGDM_LR.get_value() * EPOCH_LR_COEFF)
                RECURR_SGDM_LR.set_value(RECURR_SGDM_LR.get_value() * EPOCH_LR_COEFF)
            ADAM_EPOCHS = 0
        else:
            for _ in xrange(max_epoch):
                RESNET_ADAM_LR.set_value(RESNET_ADAM_LR.get_value() * EPOCH_LR_COEFF)
                RECURR_ADAM_LR.set_value(RECURR_ADAM_LR.get_value() * EPOCH_LR_COEFF)
        NUM_EPOCHS -= max_epoch
        param_values_file = 'ln_hs_param_values_{}.pkl'.format(max_epoch)

    logger.info('Building the network.')
    im_features = lasagne.layers.get_output(resnet['pool5'])
    im_features = T.flatten(im_features, outdim=2) # batch size, number of features
    cap_out_var = T.imatrix('cap_out')  # batch size, seq len
    cap_in_var = T.imatrix('cap_in')    # batch size, seq len
    mask_var = T.bmatrix('mask_var')    # batch size, seq len
    gate = lasagne.layers.Gate(W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
                               W_cell=lasagne.init.Normal(), b=lasagne.init.Constant(0.0))
    cell_gate = lasagne.layers.Gate(W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
                                    W_cell=None, b=lasagne.init.Constant(0.0),
                                    nonlinearity=lasagne.nonlinearities.tanh)
    forget_gate = lasagne.layers.Gate(W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
                                      W_cell=lasagne.init.Normal(), b=lasagne.init.Constant(5.0))
    l_in = lasagne.layers.InputLayer((None, None), cap_in_var, name="l_in")
    l_mask = lasagne.layers.InputLayer((None, None), mask_var, name="l_mask")
    l_hid = lasagne.layers.InputLayer((None, HIDDEN_SIZE), input_var=im_features, name="l_hid")
    l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=WORD_SIZE, output_size=EMBEDDING_SIZE, name="l_emb")
    l_lstm = LNLSTMLayer(l_emb, HIDDEN_SIZE, ingate=gate, forgetgate=forget_gate, cell=cell_gate,
                                    outgate=gate, hid_init=l_hid, peepholes=True, grad_clipping=RNN_GRAD_CLIP,
                                    mask_input=l_mask, precompute_input=False, name="l_lstm") # batch size, seq len, hidden size
    l_reshape = lasagne.layers.ReshapeLayer(l_lstm, (-1, [2]), name="l_reshape") # batch size * seq len, hidden size
    l_fc = lasagne.layers.DenseLayer(l_reshape, DENSE_SIZE, b=lasagne.init.Constant(5.0),
                                    nonlinearity=lasagne.nonlinearities.rectify, name="l_fc")
    l_drp = lasagne.layers.DropoutLayer(l_fc, 0.2, name="l_drp")
    l_hs = HierarchicalSoftmaxLayer(l_drp, WORD_SIZE, name="l_hs") # batch size * seq len, WORD SIZE
    l_slice = lasagne.layers.SliceLayer(l_lstm, -1, axis=1, name="l_slice")

    if CONTINUE:
        logger.info('Setting model weights from epoch {}'.format(max_epoch))
        param_values = pickle.load(open(param_values_file, 'rb'))
        lasagne.layers.set_all_param_values(l_hs, param_values['recurrent'])
        lasagne.layers.set_all_param_values(resnet['pool5'], param_values['resnet'])

    logger.info('Creating output and loss variables')
    prediction = lasagne.layers.get_output(l_hs, deterministic=False)
    flat_cap_out_var = T.flatten(cap_out_var, outdim=1)
    loss = T.sum(lasagne.objectives.categorical_crossentropy(prediction, flat_cap_out_var))
    caption_features = lasagne.layers.get_output(l_slice, deterministic=False)
    order_embedding_loss = T.pow(T.maximum(0, caption_features - im_features), 2).sum()
    total_loss = loss + ORDER_VIOLATION_COEFF * order_embedding_loss

    deterministic_prediction = lasagne.layers.get_output(l_hs, deterministic=True)
    deterministic_captions = lasagne.layers.get_output(l_slice, deterministic=True)
    deterministic_loss = T.sum(lasagne.objectives.categorical_crossentropy(deterministic_prediction, flat_cap_out_var))
    deterministic_order_embedding_loss = T.pow(T.maximum(0, deterministic_captions - im_features), 2).sum()
    deterministic_total_loss = deterministic_loss + ORDER_VIOLATION_COEFF * deterministic_order_embedding_loss

    logger.info('Getting all parameters and creating update rules.')
    resnet_params = lasagne.layers.get_all_params(resnet['pool5'], trainable=True)
    recurrent_params = lasagne.layers.get_all_params(l_hs, trainable=True)
    resnet_grads = T.grad(total_loss, resnet_params)
    recurrent_grads = T.grad(total_loss, recurrent_params)
    resnet_grads = [T.clip(g, -TOTAL_GRAD_CLIP, TOTAL_GRAD_CLIP) for g in resnet_grads]
    recurrent_grads = [T.clip(g, -TOTAL_GRAD_CLIP, TOTAL_GRAD_CLIP) for g in recurrent_grads]
    resnet_grads, resnet_norm = lasagne.updates.total_norm_constraint(resnet_grads, TOTAL_MAX_NORM, return_norm=True)
    recurrent_grads, recurrent_norm = lasagne.updates.total_norm_constraint(recurrent_grads, TOTAL_MAX_NORM, return_norm=True)
    resnet_adam = lasagne.updates.adam(resnet_grads, resnet_params, learning_rate=RESNET_ADAM_LR)
    recurrent_adam = lasagne.updates.adam(recurrent_grads, recurrent_params, learning_rate=RECURR_ADAM_LR)
    resnet_adam_items = resnet_adam.items()
    recurrent_adam_items = recurrent_adam.items()
    resnet_adam_items.extend(recurrent_adam_items)
    adam_updates = OrderedDict(resnet_adam_items)

    logger.info("Creating the Adam update Theano function")
    adam_train_fun = theano.function([resnet['input'].input_var, cap_in_var, mask_var, cap_out_var],
                                     [total_loss, order_embedding_loss, resnet_norm, recurrent_norm],
                                     updates=adam_updates)

    resnet_sgdm = lasagne.updates.nesterov_momentum(resnet_grads, resnet_params, learning_rate=RESNET_SGDM_LR, momentum=0.8)
    recurrent_sgdm = lasagne.updates.nesterov_momentum(recurrent_grads, recurrent_params, learning_rate=RECURR_SGDM_LR, momentum=0.9)
    resnet_sgdm_items = resnet_sgdm.items()
    recurrent_sgdm_items = recurrent_sgdm.items()
    resnet_sgdm_items.extend(recurrent_sgdm_items)
    sgdm_updates = OrderedDict(resnet_sgdm_items)

    logger.info("Creating the SGDM update Theano function")
    sgdm_train_fun = theano.function([resnet['input'].input_var, cap_in_var, mask_var, cap_out_var],
                                     [total_loss, order_embedding_loss, resnet_norm, recurrent_norm],
                                     updates=sgdm_updates)
    logger.info("Creating the evaluation Theano function")
    eval_fun = theano.function([resnet['input'].input_var, cap_in_var, mask_var, cap_out_var],
                               [deterministic_total_loss, deterministic_order_embedding_loss])

    logger.info('Loading the COCO Captions training and validation sets.')
    coco_train = COCOCaptionDataset(train_images_path, train_annotations_filepath, train_buckets,
                                    bucket_minibatch_sizes, word2idx, mean_im, True)
    coco_valid = COCOCaptionDataset(valid_images_path, valid_annotations_filepath, valid_buckets,
                                    bucket_minibatch_sizes, word2idx, mean_im, False)

    deleted_adam_resources = False
    total_loss_values = {}
    order_embedding_loss_values = {}
    resnet_norm_values = {}
    recurrent_norm_values = {}
    det_total_loss_values = {}
    det_order_embedding_loss_values = {}

    logger.info("Starting the training process...")
    START = 1
    if CONTINUE:
        START = max_epoch + 1
    for e in xrange(START, NUM_EPOCHS + 1):
        logger.info("Starting epoch".format(e))

        total_loss_values[e] = []
        order_embedding_loss_values[e] = []
        resnet_norm_values[e] = []
        recurrent_norm_values[e] = []
        det_total_loss_values[e] = []
        det_order_embedding_loss_values[e] = []

        if e <= ADAM_EPOCHS:
            train_fun = adam_train_fun
            RESNET_ADAM_LR.set_value(np.float32(RESNET_ADAM_LR.get_value() * EPOCH_LR_COEFF))
            RECURR_ADAM_LR.set_value(np.float32(RECURR_ADAM_LR.get_value() * EPOCH_LR_COEFF))
            logger.info("Training with the Adam update function")
            logger.info("ResNet LR: {}, Recurrent LR: {}".format(RESNET_ADAM_LR.get_value(), RECURR_ADAM_LR.get_value()))
        else:
            if not deleted_adam_resources:
                deleted_adam_resources = True
                logger.info("Deleting Adam training function resources")
                del resnet_adam_items
                del adam_updates
                del adam_train_fun
                logger.info("Calling gc.collect() to remove resource from GC")
                while gc.collect() > 0:
                    pass
            train_fun = sgdm_train_fun
            RESNET_SGDM_LR.set_value(np.float32(RESNET_SGDM_LR.get_value() * EPOCH_LR_COEFF))
            RECURR_SGDM_LR.set_value(np.float32(RECURR_SGDM_LR.get_value() * EPOCH_LR_COEFF))
            logger.info("Training with the SGDM update function")
            logger.info("ResNet LR: {}, Recurrent LR: {}".format(RESNET_SGDM_LR.get_value(), RECURR_SGDM_LR.get_value()))
        mb = 0
        now = datetime.now()
        for im, cap_in, cap_out in coco_train:
            tl, oe, resn, recn = train_fun(im, cap_in, (cap_in > 0).astype(np.int8), cap_out)
            logger.debug("Epoch: {}, Minibatch: {}, Total Loss: {}, Order-embedding loss: {}, ResNet norm: {}, Recurrent norm: {}".format(e, mb, tl, oe, resn, recn))
            total_loss_values[e].append(tl)
            order_embedding_loss_values[e].append(oe)
            resnet_norm_values[e].append(resn)
            recurrent_norm_values[e].append(recn)
            mb += 1
        logger.info("Training epoch {} took {}.".format(e, datetime.now() - now))
        logger.info("Epoch {} results".format(e))
        logger.info("Mean total loss: {}".format(np.mean(total_loss_values[e])))
        logger.info("Mean order embedding loss: {}".format(np.mean(order_embedding_loss_values[e])))
        logger.info("Mean ResNet norm: {}".format(np.mean(resnet_norm_values[e])))
        logger.info("Mean Recurrent norm: {}".format(np.mean(recurrent_norm_values[e])))

        logger.info("Saving model parameters for epoch {}".format(e))
        pickle.dump({'resnet':lasagne.layers.get_all_param_values(resnet['pool5']),
                     'recurrent':lasagne.layers.get_all_param_values(l_hs)},
                      open('ln_hs_param_values_{}.pkl'.format(e), 'wb'), protocol=-1)
        logger.info("Saving loss values for epoch {}".format(e))
        pickle.dump({'total loss':total_loss_values, 'oe loss': order_embedding_loss_values,
                     'resnet norm': resnet_norm_values, 'recurrent norm': recurrent_norm_values},
                     open('ln_hs_training_losses.pkl', 'wb'), protocol=-1)

        if e % 2 == 0:
            logger.info("Evaluating the model on the validation set.")
            mb = 0
            for im, cap_in, cap_out in coco_valid:
                tl, oe = eval_fun(im, cap_in, (cap_in > 0).astype(np.int8), cap_out)
                logger.debug("Epoch: {}, Minibatch: {}, Validation total loss: {}, Validation order-embedding loss: {}".format(e, mb, tl, oe))
                det_total_loss_values[e].append(tl)
                det_order_embedding_loss_values[e].append(oe)
                mb += 1
            logger.info("Epoch {} validation results".format(e))
            logger.info("Mean validation total loss: {}".format(np.mean(det_total_loss_values[e])))
            logger.info("Mean validation order-embedding loss: {}".format(np.mean(det_order_embedding_loss_values[e])))

            logger.info("Saving validation loss values for epoch {}".format(e))
            pickle.dump({'total loss': det_total_loss_values, 'oe loss': det_order_embedding_loss_values},
                        open('ln_hs_validation_losses.pkl', 'wb'), protocol=-1)
