def init_models(num_classes, max_sentence, device):
    return [
        #{
        #    'model'     : 'DeepPavlov/rubert-base-cased-sentence',
        #    'n_classes' : num_classes,
        #    'max_len'   : max_sentence,
        #    'device'    : device,
        #    'tokenizer' : 'DeepPavlov/rubert-base-cased-sentence',
        #    'criterion' : 'CrossEntropyLoss',
        #    'optimizer' : 'AdamW',
        #    'name'      : 'DeepPavlov'
        #},
        #{
        #    'model'     : 'sentence-transformers/LaBSE',
        #    'n_classes' : num_classes,
        #    'max_len'   : max_sentence,
        #    'device'    : device,
        #    'tokenizer' : 'sentence-transformers/LaBSE',
        #    'criterion' : 'CrossEntropyLoss',
        #    'optimizer' : 'AdamW',
        #    'name'      : 'LaBSE'
        #},
        #{  # На RTX 4060 16Gb эту модель запустить нельзя
        #    'model'     : 'albert/albert-xxlarge-v2',
        #    'n_classes' : num_classes,
        #    'max_len'   : max_sentence,
        #    'device'    : device,
        #    'tokenizer' : 'albert/albert-xxlarge-v2',
        #    'criterion' : 'CrossEntropyLoss',
        #    'optimizer' : 'AdamW',
        #    'name'      : 'AlbertXXlargeV2'
        #},
        {
            'model'     : 'microsoft/deberta-base-mnli',
            'n_classes' : num_classes,
            'max_len'   : max_sentence,
            'device'    : device,
            'tokenizer' : 'microsoft/deberta-base-mnli',
            'criterion' : 'CrossEntropyLoss',
            'optimizer' : 'AdamW',
            'name'      : 'DeBertaBase'
        }
    ]
