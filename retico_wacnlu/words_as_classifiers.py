"""A module for grounded semantics"""

# retico
import retico_core
from retico_vision.vision import ObjectFeaturesIU
from retico_core.text import SpeechRecognitionIU
from retico_wacnlu.wac import WAC
from retico_wacnlu.common import GroundedFrameIU

from collections import deque
import numpy as np
import os.path as osp
from tqdm import tqdm
import threading

class WordsAsClassifiersModule(retico_core.AbstractModule):
    """A model of grounded semantics. 

    Attributes:
        model_dir(str): directory where WAC classifiers are saved
    """

    @staticmethod
    def name():
        return "SLIM Group WAC Model"

    @staticmethod
    def description():
        return "WAC Visually-Grounded Model"

    @staticmethod
    def input_ius():
        return [ObjectFeaturesIU,SpeechRecognitionIU]

    @staticmethod
    def output_iu():
        return GroundedFrameIU

    def __init__(self, wac_dir, train_mode=False, **kwargs):
        """Loads the WAC models.

        Args:
            model_dir (str): The path to the directory of WAC classifiers saved as pkl files.
        """
        super().__init__(**kwargs)
        self.wac = WAC(wac_dir)
        self.word_buffer = None
        self.itts = 0
        self.train_mode = train_mode
        self.queue = deque(maxlen=1)

    def train_wac(self):
        wac = self.wac.copy()
        print('updating negatives')
        wac.create_negatives()
        print('training')
        wac.train()
        print('persisting')
        wac.persist_model()


    def process_update(self, update_message):
        # print("WAC process update")
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            frame = {}
            # print(iu)
            if isinstance(iu, SpeechRecognitionIU):
                print(iu.get_text())
                if iu.get_text() == '': continue
                self.word_buffer = iu.get_text().lower().split()[-1]
                if not self.train_mode:
                    frame['word_to_find'] = self.word_buffer
            
            # when new objects are observed (i.e., not SpeechRecognitionIUs)
            if isinstance(iu, ObjectFeaturesIU):
                objects = iu.payload
                if objects is None: return
                # WAC wants a list of intents (objectIDs) and their corresponding features in a tuple
                if len(objects) == 0: return
                intents = objects.keys()
                features = [np.array(objects[obj_id]) for obj_id in objects]
                
                if not self.train_mode:
                    word,_ = self.wac.best_word((intents, features[0]))
                    print("best word for object:", word)
                    frame['best_known_word'] = word
                
                if self.train_mode:
                    if self.word_buffer is not None:
                        self.wac.add_positive_observation(self.word_buffer, features[0])
                        self.itts += 1
                        if self.itts % 2 == 0:
                            t = threading.Thread(target=self.train_wac)
                            t.start()

                if self.word_buffer is not None:
                    target = self.wac.best_object(self.word_buffer, (intents, features))
                    if target is not None: 
                        print('best object', target)
                        frame['best_object'] = target[0] 
                        frame['obj_confidence'] = target[1] 

            if len(frame) == 0: return
            output_iu = self.create_iu(iu)
            output_iu.set_frame(frame)
            output_iu = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
            self.append(output_iu)
            

    def prepare_run(self):
        if not self.train_mode:
            self.wac.load_model()
