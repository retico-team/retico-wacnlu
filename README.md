# retico-wacnlu
Words-as-Classifiers model as reference resolution / NLU module

### Installation and Requirements

### Example

```python
import sys
from retico import *

prefix = '/path/to/modules/'
sys.path.append(prefix+'retico-clip')
sys.path.append(prefix+'retico-vision')
sys.path.append(prefix+'retico-yolov8')

# from retico_yolov8 import YoloV8
from retico_core.audio import MicrophoneModule
from retico_googleasr.googleasr import GoogleASRModule
from retico_core.text import IncrementalizeASRModule
from retico_clip.clip import ClipObjectFeatures
from retico_vision.vision import WebcamModule 
from retico_yolov8.yolov8 import Yolov8
from retico_wacnlu.words_as_classifiers import 

wac_dir = 'path/to/wac/classifiers'

mic = MicrophoneModule(1000)
asr = GoogleASRModule()
iasr = IncrementalizeASRModule()
webcam = WebcamModule()
yolo = Yolov8()
feats = ClipObjectFeatures(show=True)
wac = WordsAsClassifiersModule(wac_dir=wac_dir)
debug = modules.DebugModule()

webcam.subscribe(yolo)
yolo.subscribe(feats)
feats.subscribe(wac)
asr.subscribe(iasr)
iasr.subscribe(wac)
wac.subscribe(debug)

mic.run()
asr.run()
wac.run()
iasr.run()
debug.run()

input()

mic.stop()
asr.stop()
iasr.stop()
wac.stop()
debug.stop()
```

### Citation

```

@inproceedings{kennington-schlangen-2015-simple,
    title = "Simple Learning and Compositional Application of Perceptually Grounded Word Meanings for Incremental Reference Resolution",
    author = "Kennington, Casey  and
      Schlangen, David",
    booktitle = "Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = jul,
    year = "2015",
    address = "Beijing, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P15-1029",
    doi = "10.3115/v1/P15-1029",
    pages = "292--301",
}

```