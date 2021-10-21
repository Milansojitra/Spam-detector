from numpy.lib.function_base import piecewise
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from fastapi import FastAPI

app = FastAPI() 

loaded_model = load_model('spam_model.h5')

def get_token(texts):
    MAX_NUM_WORDS=280
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    return  pad_sequences(sequences, maxlen=MAX_NUM_WORDS)
     
    
@app.get('/predict/{text}')  
def message(text:str): 
    token = get_token([text])
    label = np.argmax(loaded_model.predict(token)[0])

    print(f"{text} : {label}")
    
    return str(label)

if __name__ == '__main__':  
    app.run(debug = True)  

print(loaded_model.predict(get_token(['you win get iphone call on 1234567']))[0])
print(loaded_model.predict(get_token(['your registration done successfully']))[0])
    