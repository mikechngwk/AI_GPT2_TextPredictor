import torch
from fastapi import APIRouter, Depends
from app.model import get_model_and_tokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

router = APIRouter()


def predict_next_sentence(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, input_text: str):
    """
    Predict the next sentence based on input text using the preloaded model and tokenizer.
    1. Encode text to token
    2. Generate the predicted sentence based on input
    3. Decode token back to text
    """

    inputs = tokenizer.encode(input_text, return_tensors='pt')

    with torch.no_grad():
        outputs = model.generate(inputs,
                                 max_length=len(inputs[0]) + 70,  #if you want more words, you can increase this
                                 num_return_sequences=3,  #
                                 no_repeat_ngram_size=2,  # prevent words repeat
                                 temperature=1.0,  # the higher the more random~
                                 top_k=50,  # so all words are converted to token, so 50 it means next word chosen is from top 50 most likely words
                                 top_p=0.95,  # model can sample from larger set of tokens (more interesting output)
                                 do_sample=True)  # use sampling to generate diverse text

    # Decode the generated tokens back into text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)


    return generated_text[len(input_text):]


@router.get("/predict_next_sentence/")
def get_next_sentence(input_text: str, model_and_tokenizer=Depends(get_model_and_tokenizer)):
    model, tokenizer = model_and_tokenizer  # Unpack the model and tokenizer <-- dependency injection just like previous project
    prediction = predict_next_sentence(model, tokenizer, input_text)
    return {"predicted_sentence": prediction}
