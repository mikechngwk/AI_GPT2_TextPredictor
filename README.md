# **AI_GPT2_Next_Sentence_Prediction**

## **Project Description**

This project demonstrates the use of machine learning (specifically GPT-2, a transformer-based model) to predict the next sentence based on user input. The project uses FastAPI to serve an API that can take a text input and generate the next logical sentence based on the input, simulating a conversation or text continuation.

The model leverages the GPT-2 model, which has been pre-trained on vast text data, allowing it to generate human-like text. This project demonstrates how to set up a simple machine learning-based web service using FastAPI and Hugging Face's `transformers` library.

## **Technologies Used**

- **Python**: The primary programming language for building the application.
- **FastAPI**: A modern web framework for building APIs quickly with Python.
- **Uvicorn**: ASGI server used to run the FastAPI application.
- **Transformers**: A library from Hugging Face to interact with pre-trained NLP models like GPT-2.
- **PyTorch**: A machine learning framework used by GPT-2 for tensor operations.
- **torch.no_grad()**: Used to prevent tracking of gradients during inference to save memory.

## **Installation Instructions**

To run the project, you'll need to install the following dependencies. You can set up your environment and install the required packages using the `requirements.txt` file.

### 1. Set up your virtual environment (optional but recommended):

```bash
python -m venv venv
```

### 2. Activate the virtual environment:

- On **Windows**:
```bash
venv\Scripts\activate
```

- On **macOS/Linux**:
```bash
source venv/bin/activate
```
### 3. Install the dependencies:
Once the virtual environment is activated, run the following command to install all the required packages:
```bash
pip install -r requirements.txt
```

### 4. Running the Application:
After installing the dependencies, you can run the FastAPI server with the following command:
```bash
uvicorn app.main:app --reload
```
## **How It Works**

1. **Model and Tokenizer Loading**:
   - The GPT-2 model and tokenizer are loaded in the `app.model` module.
   - The model is used to generate the next sentence based on the input text.

2. **API Endpoint**:
   - The API endpoint `/predict_next_sentence/` accepts user input text and generates the next sentence using the pre-trained GPT-2 model.
   - The prediction is returned in the response with the generated sentence.

3. **Text Generation**:
   - The `predict_next_sentence` function uses the GPT-2 model to generate text.
   - The function utilizes the `model.generate()` method, which produces a continuation of the input text based on a sampling process with configurable randomness and diversity.

## **API Example**

- **URL**: `/predict_next_sentence/`
- **Method**: `GET`
- **Query Parameter**: `input_text` (Text to base the prediction on)

## **Sample Request:**
```bash
GET /predict_next_sentence/?input_text=Lebron%20James%20is%20the%20greatest%20of%20all%20time
```
## **Sample Response:**
```bash
{
  "predicted_sentence": " in basketball.\n\nThat's why he's the best. Not in the sense of stats or awards or anything. He's good in that he makes the game more fun and exciting, which in turn excites me to no end. Like all greats, he takes the love and adoration and praise and love for him to the next level"
}
```
## **Example Input/Output:**
- **Input**:
  - **Text**: "Lebron James is the greatest of all time"
- **Output**:
  - **predicted_sentence**: "in basketball.\n\nThat's why he's the best. Not in the sense of stats or awards or anything. He's good in that he makes the game more fun and exciting, which in turn excites me to no end. Like all greats, he takes the love and adoration and praise and love for him to the next level"

## **Code Overview**

- **app/main.py**: This is the entry point to the FastAPI app. It initializes the API and defines the route for predicting the next sentence.
- **app/model.py**: This module handles loading the model and tokenizer. It provides a `get_model_and_tokenizer` function to inject dependencies into the FastAPI route.
- **app/routers/predict.py**: Contains the logic for the `/predict_next_sentence/` endpoint, where the input text is processed and passed to the model for prediction.
- **requirements.txt**: Lists the dependencies needed for the project. Use this file to install the required libraries.










