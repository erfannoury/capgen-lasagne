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
from datetime import datetime
from collections import OrderedDict
from mscoco_threaded_iter import COCOCaptionDataset
sys.path.append('/home/noury/codevault/Recipes/modelzoo/')
sys.path.append('/home/noury/codevault/seq2seq-lasagne/')
from resnet50 import build_model
from CustomLSTMLayer import LNLSTMLayer
from HierarchicalSoftmax import HierarchicalSoftmaxLayer
from LayerNormalization import LayerNormalizationLayer
sys.setrecursionlimit(10000)

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(message)s', '%m/%d/%Y %I:%M:%S %p')
    fh = logging.FileHandler('showntell_noft_ln_hs.log')
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

    bucket_minibatch_sizes = {16:256, 32:128, 64:64}


    logger.info('Creating global variables')
    CONTINUE = False
    HIDDEN_SIZE = 2048
    EMBEDDING_SIZE = 300
    WORD_SIZE = len(idx2word)
    DENSE_SIZE = 1024
    L2_COEFF = 1e-3
    RNN_GRAD_CLIP = 64
    TOTAL_MAX_NORM = 128
    RECURR_LR = theano.shared(np.float32(0.001), 'recurrent lr')
    EPOCH_LR_COEFF = np.float32(0.5)
    NUM_EPOCHS = 15
    validation_losses = []
    total_loss_values = []
    l2_values = []
    recurrent_norm_values = []
    validation_total_loss_values = []
    validation_l2_values = []

    logger.info('Building the network.')
    im_features = lasagne.layers.get_output(resnet['pool5'])
    im_features = T.flatten(im_features, outdim=2) # batch size, number of features 
    cap_out_var = T.imatrix('cap_out')  # batch size, seq len
    cap_in_var = T.imatrix('cap_in')    # batch size, seq len
    mask_var = T.bmatrix('mask_var')    # batch size, seq len
    
    gate = lasagne.layers.Gate(W_in=lasagne.init.Normal(0.02), W_hid=lasagne.init.Orthogonal(),
                               W_cell=lasagne.init.Normal(), b=lasagne.init.Constant(0.0))
    cell_gate = lasagne.layers.Gate(W_in=lasagne.init.Normal(0.02), W_hid=lasagne.init.Orthogonal(),
                                    W_cell=None, b=lasagne.init.Constant(0.0),
                                    nonlinearity=lasagne.nonlinearities.tanh)
    forget_gate = lasagne.layers.Gate(W_in=lasagne.init.Normal(0.02), W_hid=lasagne.init.Orthogonal(),
                                      W_cell=lasagne.init.Normal(), b=lasagne.init.Constant(5.0))

    # Image embedding layer
    l_im_in = lasagne.layers.InputLayer((None, HIDDEN_SIZE), input_var=im_features, name="l_im_in")
    l_im_emb = lasagne.layers.DenseLayer(l_im_in, EMBEDDING_SIZE, b=lasagne.init.Constant(1.0), name="l_im_emb")
    l_im_reshape = lasagne.layers.ReshapeLayer(l_im_emb, ([0], 1, [1]), name="l_im_reshape")
    l_im_mask = lasagne.layers.InputLayer((None, 1), input_var=T.ones((im_features.shape[0], 1), dtype=np.float32))

    # Caption embedding layer
    l_in = lasagne.layers.InputLayer((None, None), cap_in_var, name="l_in")
    l_mask = lasagne.layers.InputLayer((None, None), mask_var, name="l_mask")
    l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=WORD_SIZE, output_size=EMBEDDING_SIZE, name="l_emb")

    # Sequence and mask concatentation
    l_concat_embs = lasagne.layers.ConcatLayer([l_im_reshape, l_in], axis=1, name="l_concat_embs")
    l_concat_masks = lasagne.layers.ConcatLayer([l_im_mask, l_mask], axis=1, name="l_concat_masks")

    # LSTM layer and the rest
    l_lstm = LNLSTMLayer(l_concat_embs, HIDDEN_SIZE, ingate=gate, forgetgate=forget_gate, cell=cell_gate,
                                    outgate=gate, peepholes=False, grad_clipping=RNN_GRAD_CLIP,
                                    mask_input=l_concat_masks, precompute_input=False,
                                    alpha_init=lasagne.init.Constant(0.1), # as suggested by Ryan Kiros on Twitter
                                    normalize_cell=False,
                                    name="l_lstm") # batch size, seq len, hidden size
    l_reshape = lasagne.layers.ReshapeLayer(l_lstm, (-1, [2]), name="l_reshape") # batch size * seq len, hidden size
    l_fc = lasagne.layers.DenseLayer(l_reshape, DENSE_SIZE, b=lasagne.init.Constant(5.0),
                                    nonlinearity=lasagne.nonlinearities.rectify, name="l_fc")
    l_drp = lasagne.layers.DropoutLayer(l_fc, 0.3, name="l_drp")
    l_hs = HierarchicalSoftmaxLayer(l_drp, WORD_SIZE, name="l_hs") # batch size * seq len, WORD SIZE
    l_slice = lasagne.layers.SliceLayer(l_lstm, -1, axis=1, name="l_slice")

    if CONTINUE:
        import glob
        param_values = glob.glob('showntell_noft_ln_hs_param_values_*.pkl')
        max_epoch = max(map(lambda x: int(x[len('showntell_noft_ln_hs_param_values_'):-len('.pkl')]), param_values))
        logger.info('Continue training from epoch {}'.format(max_epoch + 1))
        logger.info('Setting previous parameter values from epoch {}'.format(max_epoch))

        logger.info('Setting model weights from epoch {}'.format(max_epoch))
        param_values_file = 'showntell_noft_ln_hs_param_values_{}.pkl'.format(max_epoch)
        param_values = pickle.load(open(param_values_file, 'rb'))
        lasagne.layers.set_all_param_values(l_hs, param_values['recurrent'])
        lasagne.layers.set_all_param_values(resnet['pool5'], param_values['resnet'])
        RECURR_LR = theano.shared(np.float32(param_values['lr']), 'recurrent lr')

        [total_loss_values, l2_values,
         recurrent_norm_values]= pickle.load(open('showntell_noft_ln_hs_training_losses.pkl', 'rb'))

        [validation_total_loss_values,
         validation_l2_values] = pickle.load(open('showntell_noft_ln_hs_validation_losses.pkl', 'rb'))

        [validation_losses, recurr_lr_val] = pickle.load(open('showntell_noft_ln_hs_artifacts.pkl', 'rb'))

    logger.info('Creating output and loss variables')
    prediction = lasagne.layers.get_output(l_hs, deterministic=False)
    flat_cap_out_var = T.flatten(cap_out_var, outdim=1)
    flat_mask_var = T.flatten(lasagne.layers.get_output(l_mask), outdim=1)
    loss = T.mean(lasagne.objectives.categorical_crossentropy(prediction, flat_cap_out_var)[flat_mask_var.nonzero()])
    l2 = lasagne.regularization.regularize_network_params(l_hs, lasagne.regularization.l2)
    total_loss = loss + L2_COEFF * l2

    deterministic_prediction = lasagne.layers.get_output(l_hs, deterministic=True)
    deterministic_loss = T.mean(lasagne.objectives.categorical_crossentropy(deterministic_prediction, flat_cap_out_var)[flat_mask_var.nonzero()])
    deterministic_l2 = lasagne.regularization.regularize_network_params(l_hs, lasagne.regularization.l2)
    deterministic_total_loss = deterministic_loss + L2_COEFF * deterministic_l2

    logger.info('Getting all parameters and creating update rules.')
    recurrent_params = lasagne.layers.get_all_params(l_hs, trainable=True)
    recurrent_grads = T.grad(total_loss, recurrent_params)
    recurrent_grads, recurrent_norm = lasagne.updates.total_norm_constraint(recurrent_grads, TOTAL_MAX_NORM, return_norm=True)
    recurrent_updates = lasagne.updates.rmsprop(recurrent_grads, recurrent_params, learning_rate=RECURR_LR)

    logger.info("Creating the Theano function for Adam update")
    train_fun = theano.function([resnet['input'].input_var, cap_in_var, mask_var, cap_out_var],
                                     [total_loss, l2, recurrent_norm],
                                     updates=recurrent_updates)

    logger.info("Creating the evaluation Theano function")
    eval_fun = theano.function([resnet['input'].input_var, cap_in_var, mask_var, cap_out_var],
                               [deterministic_total_loss, deterministic_l2])

    logger.info('Loading the COCO Captions training and validation sets.')
    coco_train = COCOCaptionDataset(train_images_path, train_annotations_filepath, train_buckets,
                                    bucket_minibatch_sizes, word2idx, mean_im, True)
    coco_valid = COCOCaptionDataset(valid_images_path, valid_annotations_filepath, valid_buckets,
                                    bucket_minibatch_sizes, word2idx, mean_im, False)

    logger.info("Starting the training process...")
    START = 1
    if CONTINUE:
        START = max_epoch + 1
    for e in xrange(START, NUM_EPOCHS + 1):
        logger.info("Starting epoch".format(e))

        if len(validation_losses) > 2 and \
           validation_losses[-3] < validation_losses[-1] and \
           validation_losses[-2] < validation_losses[-1]:
            RECURR_LR.set_value(RECURR_LR.get_value() * EPOCH_LR_COEFF)
            logger.info("Lowering the learning rate to {}".format(RECURR_LR.get_value()))

        logger.info("Starting training on epoch {} with LR = {}".format(e, RECURR_LR.get_value()))
        mb = 0
        now = datetime.now()
        for im, cap_in, cap_out in coco_train:
            tl, el2, recn = train_fun(im, cap_in, (cap_in > 0).astype(np.int8), cap_out)
            logger.debug("Epoch: {}, Minibatch: {}, Total Loss: {}, L2 value: {}, Recurrent norm: {}".format(e, mb, tl, el2, recn))
            total_loss_values.append(tl)
            l2_values.append(el2)
            recurrent_norm_values.append(recn)
            mb += 1

        logger.info("Training epoch {} took {}.".format(e, datetime.now() - now))
        logger.info("Epoch {} results:".format(e))
        logger.info("\t\tMean total loss: {}".format(np.mean(total_loss_values[-mb:])))
        logger.info("\t\tMean l2 value: {}".format(np.mean(l2_values[-mb:])))
        logger.info("\t\tMean Recurrent norm: {}".format(np.mean(recurrent_norm_values[-mb:])))

        logger.info("Saving model parameters for epoch {}".format(e))
        pickle.dump({'resnet':lasagne.layers.get_all_param_values(resnet['pool5']),
                     'recurrent':lasagne.layers.get_all_param_values(l_hs),
                     'mean image':mean_im,
                     'lr':RECURR_LR.get_value()},
                      open('showntell_noft_ln_hs_param_values_{}.pkl'.format(e), 'wb'), protocol=-1)
        logger.info("Saving loss values for epoch {}".format(e))
        pickle.dump([total_loss_values, l2_values,
                     recurrent_norm_values],
                     open('showntell_noft_ln_hs_training_losses.pkl', 'wb'), protocol=-1)

        logger.info("Validating the model on epoch {} on the validation set.".format(e))
        mb = 0
        now = datetime.now()
        for im, cap_in, cap_out in coco_valid:
            tl, el2 = eval_fun(im, cap_in, (cap_in > 0).astype(np.int8), cap_out)
            logger.debug("Validation epoch: {}, Minibatch: {}, Validation total loss: {}, Validation l2 value: {}".format(e, mb, tl, el2))
            validation_total_loss_values.append(tl)
            validation_l2_values.append(el2)
            mb += 1
        logger.info("Validating epoch {} took {}.".format(e, datetime.now() - now))
        logger.info("Epoch {} validation results:".format(e))
        logger.info("\t\tValidation mean total loss: {}".format(np.mean(validation_total_loss_values[-mb:])))
        logger.info("\t\tValidation mean l2 value: {}".format(np.mean(validation_l2_values[-mb:])))

        validation_losses.append(np.mean(validation_total_loss_values[-mb:]))

        logger.info("Saving validation loss values for epoch {}".format(e))
        pickle.dump([validation_total_loss_values, validation_l2_values],
                    open('showntell_noft_ln_hs_validation_losses.pkl', 'wb'), protocol=-1)
        pickle.dump([validation_losses, RECURR_LR.get_value()], open('showntell_noft_ln_hs_artifacts.pkl', 'wb'),
                    protocol=-1)
