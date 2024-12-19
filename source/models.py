from modelbertclassifier import ModelBertClassifier
from modelalbertclassifier import ModelAlBertClassifier
from modeldebertaclassifier import ModelDeBertaClassifier

from transformers import BertTokenizer, AlbertTokenizer, DebertaTokenizer

def init_models(num_classes, max_sentence, device):
    return [
        {
            'model'     : ModelBertClassifier(
                            model_name='bert-base-multilingual-cased',
                            n_classes=num_classes,
                            max_len=max_sentence,
                            device=device),
            'tokenizer' : BertTokenizer.from_pretrained('bert-base-multilingual-cased'),
            'criterion' : 'CrossEntropyLoss',
            'optimizer' : 'AdamW',
            'name'      : 'BertBase'
        },
        {
            'model'     : ModelBertClassifier(
                            model_name='DeepPavlov/rubert-base-cased-sentence',
                            n_classes=num_classes,
                            max_len=max_sentence,
                            device=device),
            'tokenizer' : BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-sentence'),
            'criterion' : 'CrossEntropyLoss',
            'optimizer' : 'AdamW',
            'name'      : 'DeepPavlov'
        },
                {
            'model'     : ModelBertClassifier(
                            model_name='sentence-transformers/LaBSE',
                            n_classes=num_classes,
                            max_len=max_sentence,
                            device=device),
            'tokenizer' : BertTokenizer.from_pretrained('sentence-transformers/LaBSE'),
            'criterion' : 'CrossEntropyLoss',
            'optimizer' : 'AdamW',
            'name'      : 'LaBSE'
        },
        {
            'model'     : ModelAlBertClassifier(
                            model_name='albert/albert-base-v1',
                            n_classes=num_classes,
                            max_len=max_sentence,
                            device=device),
            'tokenizer' : AlbertTokenizer.from_pretrained('albert/albert-base-v1'),
            'criterion' : 'CrossEntropyLoss',
            'optimizer' : 'AdamW',
            'name'      : 'AlbertV1'
        },
        {
            'model'     : ModelAlBertClassifier(
                            model_name='albert/albert-xxlarge-v2',
                            n_classes=num_classes,
                            max_len=max_sentence,
                            device=device),
            'tokenizer' : AlbertTokenizer.from_pretrained('albert/albert-xxlarge-v2'),
            'criterion' : 'CrossEntropyLoss',
            'optimizer' : 'AdamW',
            'name'      : 'AlbertXXlargeV2'
        },
        {
            'model'     : ModelAlBertClassifier(
                            model_name='tals/albert-xlarge-vitaminc-mnli',
                            n_classes=num_classes,
                            max_len=max_sentence,
                            device=device),
            'tokenizer' : AlbertTokenizer.from_pretrained('tals/albert-xlarge-vitaminc-mnli'),
            'criterion' : 'CrossEntropyLoss',
            'optimizer' : 'AdamW',
            'name'      : 'AlbertXVitaminc'
        },
        {
            'model'     : ModelDeBertaClassifier(
                            model_name='microsoft/deberta-base-mnli',
                            n_classes=num_classes,
                            max_len=max_sentence,
                            device=device),
            'tokenizer' : DebertaTokenizer.from_pretrained('microsoft/deberta-base-mnli'),
            'criterion' : 'CrossEntropyLoss',
            'optimizer' : 'AdamW',
            'name'      : 'DeBertaBase'
        }
    ]
