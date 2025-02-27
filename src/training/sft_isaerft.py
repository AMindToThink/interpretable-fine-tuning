# Following along with https://medium.com/myorder/fine-tuning-pythia-70m-deduped-instruction-following-llms-with-performance-evaluation-3bd0bb33b79

#%%
import os
import pandas as pd
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)

import evaluate
import nltk
from nltk.tokenize import sent_tokenize
#%%
ds = load_dataset('databricks/databricks-dolly-15k')
#%%