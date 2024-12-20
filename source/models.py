def init_models(num_classes, max_sentence, device):
    return [
        {
            'model'     : 'DeepPavlov/rubert-base-cased-sentence',
            'n_classes' : num_classes,
            'max_len'   : max_sentence,
            'device'    : device,
            'tokenizer' : 'DeepPavlov/rubert-base-cased-sentence',
            'criterion' : 'CrossEntropyLoss',
            'optimizer' : 'AdamW',
            'name'      : 'DeepPavlov'
        },
        {
            'model'     : 'sentence-transformers/LaBSE',
            'n_classes' : num_classes,
            'max_len'   : max_sentence,
            'device'    : device,
            'tokenizer' : 'sentence-transformers/LaBSE',
            'criterion' : 'CrossEntropyLoss',
            'optimizer' : 'AdamW',
            'name'      : 'LaBSE'
        },
        {
            'model'     : 'albert/albert-base-v1',
            'n_classes' : num_classes,
            'max_len'   : max_sentence,
            'device'    : device,
            'tokenizer' : 'albert/albert-base-v1',
            'criterion' : 'CrossEntropyLoss',
            'optimizer' : 'AdamW',
            'name'      : 'AlbertV1'
        },
        {
            'model'     : 'albert/albert-xxlarge-v2',
            'n_classes' : num_classes,
            'max_len'   : max_sentence,
            'device'    : device,
            'tokenizer' : 'albert/albert-xxlarge-v2',
            'criterion' : 'CrossEntropyLoss',
            'optimizer' : 'AdamW',
            'name'      : 'AlbertXXlargeV2'
        },
        {
            'model'     : 'tals/albert-xlarge-vitaminc-mnli',
            'n_classes' : num_classes,
            'max_len'   : max_sentence,
            'device'    : device,
            'tokenizer' : 'tals/albert-xlarge-vitaminc-mnli',
            'criterion' : 'CrossEntropyLoss',
            'optimizer' : 'AdamW',
            'name'      : 'AlbertXVitaminc'
        },
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
