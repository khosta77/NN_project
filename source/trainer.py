import torch
import pickle
import datetime
import warnings
import progressbar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch import device, cuda, nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from universalds import UniversalDataset

class Trainer:
    def __init__(self, models, batch_size, epochs, device):
        self.models = models
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size

    def _save_lists_to_file(self, file_name, list1, list2, list3, list4):
        with open(file_name, 'wb') as f:
            pickle.dump((list1, list2, list3, list4), f)

    def _plot(
            self,
            train_loss_epochs, test_loss_epochs,
            train_accuracy_epochs, test_accuracy_epochs,
            epochs, name
        ):
        y_max_value = max(max(train_loss_epochs), max(test_loss_epochs))
        train = [ train_loss_epochs, train_accuracy_epochs ]
        test = [ test_loss_epochs, test_accuracy_epochs ]
        label = [ 'Loss', 'Accuracy' ]
        plt.figure(figsize=(13, 4), dpi=600)
        for i in range(1, 3):
            plt.subplot(1, 2, i)
            plt.plot(train[(i - 1)], label='Train', linewidth=1.0)
            plt.plot(test[(i - 1)], label='Test', linewidth=1.0)
            plt.xlabel('Epochs')
            plt.ylabel(label[(i - 1)])
            plt.ylim([0, (y_max_value + 0.1*y_max_value) if i == 1 else 1])
            plt.xlim([0, epochs])
            plt.legend(loc=0)
            plt.grid()
        plt.savefig('img/' + name + '_metric_plot.png')

    def _optimzer(self, model):
        if model['optimizer'] == 'AdamW':
            return AdamW(model['model'].parameters(), lr=2e-5, correct_bias=False)
        name = model['name']
        print(f'\nВ модели {name} выбран не корректный optimzer, выбран AdamW\n')
        return AdamW(model['model'].parameters(), lr=2e-5, correct_bias=False)

    def _criterion(self, model):
        if model['criterion'] == 'CrossEntropyLoss':
            return nn.CrossEntropyLoss().to(self.device)
        name = model['name']
        print(f'\nВ модели {name} выбран не корректный criterion, выбран CrossEntropyLoss\n')
        return nn.CrossEntropyLoss().to(self.device)

    def _move(self, X_train, y_train, X_valid, y_valid, tokenizer):
        train_set = UniversalDataset(list(X_train), list(y_train), tokenizer)
        valid_set = UniversalDataset(list(X_valid), list(y_valid), tokenizer)

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=self.batch_size, shuffle=True)
        return train_loader, valid_loader, len(train_set), len(valid_set)

    def train_models(self, X_train, y_train, X_valid, y_valid, accurate_break=0.951):
        total_time = datetime.datetime.now()
        result = { 'model': [], 'accuracy': [], 'loss': [] }
        for i, model in enumerate(self.models):
            name = model['name']
            print('=' * 100)
            print('-' * 100)
            print('=' * 100)
            print(f'[{i + 1}/{len(self.models)}] {name}')
            train_loader, valid_loader, train_len, valid_len = self._move(X_train, y_train, X_valid, y_valid, model['tokenizer'])

            optimizer = self._optimzer(model)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, 
                                                        num_training_steps=len(train_loader) * self.epochs)

            criterion = self._criterion(model)

            classifier = model['model']

            train_accuracy_epochs, train_loss_epochs, val_accuracy_epochs, val_loss_epochs = [], [], [], []
            start_model_train_time = datetime.datetime.now()

            for epoch in range(self.epochs):
                #### Train
                classifier.fit()
                losses = []
                correct_predictions = 0
                for data in tqdm(train_loader):
                    input_ids = data["input_ids"].to(self.device)
                    attention_mask = data["attention_mask"].to(self.device)
                    targets = data["targets"].to(self.device)

                    outputs = classifier(input_ids=input_ids, attention_mask=attention_mask)

                    preds = torch.argmax(outputs.logits, dim=1)
                    loss = criterion(outputs.logits, targets)

                    correct_predictions += torch.sum(preds == targets)

                    losses.append(loss.item())
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                train_accuracy_epochs.append(float(correct_predictions.double() / train_len))
                train_loss_epochs.append(np.mean(losses))

                #### Valid
                classifier.eval()
                losses = []
                correct_predictions = 0

                with torch.no_grad():
                    for data in tqdm(valid_loader):
                        input_ids = data["input_ids"].to(self.device)
                        attention_mask = data["attention_mask"].to(self.device)
                        targets = data["targets"].to(self.device)

                        outputs = classifier(input_ids=input_ids, attention_mask=attention_mask)

                        preds = torch.argmax(outputs.logits, dim=1)
                        loss = criterion(outputs.logits, targets)
                        correct_predictions += torch.sum(preds == targets)
                        losses.append(loss.item())

                val_accuracy_epochs.append(float(correct_predictions.double() / valid_len))
                val_loss_epochs.append(np.mean(losses))

                print(
                    f'Epoch [{(epoch+1)}/{self.epochs}]: (Train/Validation) ',
                    f'Loss: {train_loss_epochs[-1]:.3f}/{val_loss_epochs[-1]:.3f}, ',
                    f'Accuracy: {train_accuracy_epochs[-1]:.3f}/{val_accuracy_epochs[-1]:.3f}, ',
                    f't: {(datetime.datetime.now() - start_model_train_time)}'
                )

                if train_accuracy_epochs[-1] >= accurate_break and val_accuracy_epochs[-1] >= accurate_break:
                    print('На обучающей и тестовой выборке достигли желаемого результата.\n',
                          'Чтобы не израходовать ресурсы машины:\t break')
                    break

                if len(val_accuracy_epochs) >= 3:
                    if val_accuracy_epochs[-3] > val_accuracy_epochs[-1]:
                        print(f'\t\t!!!Мы достигли апогея обучения!!!')
                        break

            # after learning
            self._plot(
                train_loss_epochs, val_loss_epochs,
                train_accuracy_epochs, val_accuracy_epochs,
                self.epochs, name
            )

            self._save_lists_to_file(
                ('log/' + name + '.log'),
                train_loss_epochs, val_loss_epochs,
                train_accuracy_epochs, val_accuracy_epochs,
            )

            classifier.save(name)
            result['model'].append(name)
            result['accuracy'].append(val_accuracy_epochs[-1])
            result['loss'].append(val_loss_epochs[-1])

        # after learnings models
        print('=' * 100)
        print('-' * 100)
        print('=' * 100)
        print(f'total time: {(datetime.datetime.now() - total_time)}')
        return pd.DataFrame(data=result)