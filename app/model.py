from transformers import GPT2LMHeadModel, GPT2Tokenizer
from fastapi import Depends

'''
There are alot of models like gpt2, gpt2-medium, gpt2-large.
For this project i used gpt2-large - 3.25GB
gpt2 is around 600MB <-- usee this if your internet connection is slow and ur workstation has limited memory
'''
MODEL_NAME = "gpt2-large"  # You can specify any GPT model, such as 'gpt2-medium' or 'gpt2-large'


def load_model():
    """
    Load the pre-trained GPT-2 Large model and tokenizer.
    This will be used by the prediction endpoint.
    """
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

    return model, tokenizer


def get_model_and_tokenizer(model_and_tokenizer=Depends(load_model)):
    """
    Dependency function to load the model and tokenizer when the request is made.
    Just like the previous project ML_SGDClassifier_Spam, I used dependency injection for the trained models
    """
    return model_and_tokenizer
