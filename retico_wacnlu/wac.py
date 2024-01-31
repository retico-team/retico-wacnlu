# code for WAC classifiers

import numpy as np
from sklearn import linear_model
import pickle
from operator import itemgetter
import os
from sklearn.base import clone
from tqdm import tqdm
from operator import itemgetter 
from sklearn.model_selection import train_test_split
import copy
import random
import logging

class WAC:
    
    def __init__(self,wac_dir,classifier_spec=(linear_model.LogisticRegression,{'penalty':'l2'}),compose_method='prod'):
        '''
        wac_dir: name of model for persistance and loading
        compose_method: prod, avg, sum
        
        to train:
        add observations, then call train() then persist()
        
        to evaluate:
        load() model, then call add_increment for each word in an utterance. Call new_utt() to start a new utterance. 
        '''
        self.positives = {}
        self.wac = {}
        self.trained_wac = {}
        self.model_name=wac_dir
        self.current_utt = {}
        self.current_observed_object = None
        self.current_objects = {}
        self.utt_words = []
        self.compose_method = compose_method
        self.classifier_spec = classifier_spec
        self.max_obvs = 20

    def copy(self):
        new_wac = WAC(self.model_name, self.classifier_spec, self.compose_method)
        new_wac.positives = copy.deepcopy(self.positives)
        new_wac.wac = copy.deepcopy(self.wac)
        new_wac.utt_words = list(self.utt_words)
        new_wac.trained_wac = copy.deepcopy(self.trained_wac)
        return new_wac

    def vocab(self):
        return self.trained_wac.keys()

    def set_current_observed_object(self, object_id):
        '''
        object_id is a string that represents the current object in view
        '''
        self.current_observed_object = object_id

    def associate_object(self, features):
        '''
        features are the same as for add_positive_observation (i.e., a vector) for the object in view
        '''
        if self.current_observed_object is not None:
            if self.current_observed_object not in self.current_objects:
                self.current_objects[self.current_observed_object] = list()
                self.current_objects[self.current_observed_object].append(features)
            # print("WAC: current object ids seen:", self.current_objects.keys())
            if self.max_obvs is not None and len(self.current_objects[self.current_observed_object]) < self.max_obvs:
                self.current_objects[self.current_observed_object].append(features)
    
    def add_positive_observation(self, word, features):
        if word not in self.positives: self.positives[word] = list()
        # print("LENGTH OF POSITIVE OBSERVATIONS: ", word, len(self.positives[word]))
        # if self.max_obvs is not None and len(self.positives[word]) < self.max_obvs:
        self.positives[word].append((features, 1))
        # if len(self.positives[word]) == self.max_obvs:
            # return True
        # else: 
            # return False
        # print(list(self.positives.keys()))

    def add_observation(self, word, features, label):
        if word not in self.wac: self.wac[word] = list()
        # print("WAC: adding feats", features)
        self.wac[word].append((features, label))
    
    def add_multiple_observations(self, word, features, labels):
        for f,p in zip(features,labels):
            self.add_observation(word, f, p)

    def create_negatives(self, num_negs=3):
        if len(self.positives) <= 1: return # need at least two words to find negs
        # self.wac = copy.deepcopy(self.positives) # copy the positives

        self.wac = {}
        for word in self.positives:
            l = len(self.positives[word])
            if l > self.max_obvs:
                self.wac[word] = random.sample(self.positives[word], self.max_obvs)
            else:
                self.wac[word] = copy.deepcopy(self.positives[word])

        for word in self.wac:
            negs = []
            max_len = len(self.wac[word]) * num_negs
            while len(negs) < max_len:
                for iword in self.wac:
                    if iword == word: continue
                    i = random.sample(self.wac[iword],1)[0]
                    negs.append(i[0])
            for i in negs:
                self.add_observation(word, i, 0)

    def train(self, min_obs=2):
        classifier, classf_params = self.classifier_spec
        nwac = {}
        if len(self.wac) <= 1: return
        for word in self.wac:
            if len(self.wac[word]) < min_obs: continue
            this_classf = classifier(**classf_params)
            X,y = zip(*self.wac[word])
            X = np.array(X).squeeze()
            # print(word, X.shape)
            # nsamples, nx, ny = X.shape
            # X = X.reshape((nsamples,nx*ny))
            # X = tuple(X)
            this_classf.fit(X,y)
            nwac[word] = this_classf
        self.trained_wac = nwac
    
    def load_model(self):
        self.trained_wac = {}
        existing = [f.split('.')[0] for f in os.listdir(self.model_name)]
        print('loading WAC model')
        for item in tqdm(existing):
            with open('{}/{}.pkl'.format(self.model_name, item), 'rb') as handle:
                self.trained_wac[item] = pickle.load(handle)

    def persist_model(self):
        print("TRAINED AND READY TO PERSIST", len(self.trained_wac))
        if len(self.trained_wac) == 0: return
        for word in self.trained_wac:
            with open('{}/{}.pkl'.format(self.model_name, word), 'wb') as handle:
                pickle.dump(self.trained_wac[word],handle, protocol=pickle.HIGHEST_PROTOCOL)
 
    def get_current_prediction_state(self):
        return self.current_utt
    
    def get_predicted_intent(self):
        return max(self.get_current_prediction_state(), key=itemgetter(1))

    def add_increment(self, word, context):
        predictions  = self.proba(word, context)
        self.utt_words.append(word)
        return self.compose(predictions)

    def best_word(self, context):
        if self.trained_wac is None: return None
        if len(self.trained_wac) > 0:
            probs = [(word, self.proba(word, context)) for word in self.trained_wac]
            res = max(probs, key = itemgetter(1))
            return res
        return None

    def best_object(self, word, context):
        preds =  self.proba(word, context)
        if preds is None: return None
        return max(preds, key=itemgetter(1))

    def best_object_from_memory(self, utterance):
        '''
        utterance = list of words
        '''
        self.new_utt()
        context = []
        intents = []
        if len(self.current_objects) == 0: return None
        for o in self.current_objects:
            vectors = np.array(self.current_objects[o]).squeeze()
            print('object vector length ', o, len(vectors), vectors.shape)
            context.append(np.mean(vectors, axis=0)) # use all of the stored vectors for the object, but take the mean for a "prototype"
            intents.append(o)
            print("context",o, np.array(context).shape)
        context = (intents, context)
        for word in utterance:
            self.add_increment(word, context)
        return self.get_predicted_intent()

    def add_increment(self, word, context):
        predictions  = self.proba(word, context)
        return self.compose(predictions)

    def compose(self, predictions):
        if self.current_utt == {}:
            self.current_utt = predictions 
            return self.current_utt
        
        composed = []
        for one,two in zip(predictions,self.current_utt):
            i1,p1=one
            i2,p2=two
            if i1 != i2:
                print('intent mismatch! {} != {}'.format(i1,i2))
                continue
            
            res = 0
            if self.compose_method == 'sum':
                res = p1 + p2
            if self.compose_method == 'prod':
                res = p1 * p2
            if self.compose_method == 'avg':
                res = p1 * p2 / 2.0
            
            composed.append((i1,res))
            
        self.current_utt = composed
        return self.current_utt    

    def proba(self, word, context): #use for apply, dummy id
        intents,feats = context
        if word not in self.trained_wac: return None # todo: return a distribution of all zeros?
        featsArray = np.array(feats) # .squeeze().reshape(1, -1)
        if len(featsArray.shape) > 2: featsArray = featsArray.squeeze()
        if len(featsArray.shape) < 2: featsArray = featsArray.reshape(1, -1)
        # print('feats array shape', featsArray.shape)
        # nsamples, nx, ny = featsArray.shape
        # featsArray = featsArray.reshape((nsamples,nx*ny))
        predictions = list(zip(intents,self.trained_wac[word].predict_proba(featsArray)[:,1]))
        if len(predictions) > 1: print('Predictions:', predictions)
        return predictions

    def get_current_prediction_state(self):
        return self.current_utt
    
    def get_predicted_intent(self):
        intents = self.get_current_prediction_state()
        if intents is None: return None
        return max(intents, key=itemgetter(1))        
    
    def new_utt(self):
        self.current_utt = {} 
        self.utt_words = []
    
