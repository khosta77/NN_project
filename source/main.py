import torch
import pickle
import warnings
import datetime
import progressbar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch import nn
from tqdm import tqdm
from torch import device, cuda
from sklearn.model_selection import train_test_split

from constant import PATH, NUM_CLASSES, MAX_SENTENCE, EPOCHS
from universalds import UniversalDataset
from modelbertclassifier import ModelBertClassifier
from modelalbertclassifier import ModelAlBertClassifier
from modeldebertaclassifier import ModelDeBertaClassifier
from trainer import Trainer
from models import init_models

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # Странно, если в открытом пространстве, ничего не работало
    DEVICE=device("cuda:0" if cuda.is_available() else "cpu")
    print(DEVICE)

    models = init_models(NUM_CLASSES, MAX_SENTENCE, DEVICE)

    train_data, valid_data = train_test_split(
            pd.read_csv(PATH, usecols=['annotation', 'rate'], sep=';'),
            train_size=0.9, random_state=42)
    print(f'Обучающая выборка: {len(train_data):,}\nВалидация: {len(valid_data):,}')

    trainer = Trainer(models=models, batch_size=16, epochs=EPOCHS, device=DEVICE)

    result = trainer.train_models(
        train_data['annotation'], train_data['rate'],
        valid_data['annotation'], valid_data['rate']
    )
    result.to_csv('result.csv')