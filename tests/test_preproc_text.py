import pandas as pd
from toolbox import preproc_text

def test_preproc_text():
    assert preproc_text.preprocess_text(pd.Series('This. is. a. 42 test. strings!')).values[0] \
       == pd.Series('test string').values[0]
