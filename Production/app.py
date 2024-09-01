from fastai.vision.all import *
from pathlib import Path
import gradio as gr
import torch
import pathlib
import os

def get_dog_breed(o):
    folder_name = Path(o).parent.name
    breed = str(folder_name).split('-')[1]
    return ' '.join(breed.split('_'))

model_path = Path('dog_breeds_clf_model.pkl')

on_windows = False
if os.name == 'nt':
    on_windows = True

if on_windows:
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    learn = load_learner(model_path)
    pathlib.PosixPath = temp
else:
    learn = load_learner(model_path)


categories = learn.dls.vocab
def classify_images(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

examples = ['black_labrador_retriever.jpg', 'great_dane.jpg', 'chihuahua.jpg']

intf = gr.Interface(fn=classify_images, inputs="image", outputs="label", examples=examples)
intf.launch(inline=False)