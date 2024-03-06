import json
import numpy as np
import pandas as pd

def reader(idx_doc=0, idx_tok=0):
    og_tok_id = []
    og_document = []
    og_token = []
    og_label = []

    with open("original_data/train.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

        docs = data

        for i, d in enumerate(docs):
            if i < idx_doc:
                continue
            og_token += d["tokens"][idx_tok:]
            og_label += d["labels"][idx_tok:]
            og_document += list(np.ones(len(d["tokens"]) - idx_tok) * d["document"])
            og_tok_id += range(idx_tok, len(d["tokens"]))
            if i == idx_doc and idx_tok != 0:
                idx_tok = 0
            print(f"Doc #{i} is done")
    print("Reading is done")
    

    print(len(og_document))
    print(len(og_token))
    print(len(og_label))
    print(len(og_tok_id))
    return og_document, og_token, og_label, og_tok_id

def analyzer(og_document, og_token, og_label, og_tok_id):
    row_ids = []
    documents = []
    tokens = []
    labels = []
    words = []
    for i, label in enumerate(og_label):
        if '-' in label:
            documents.append(og_document[i])
            tokens.append(og_tok_id[i])
            labels.append(label)
            words.append(og_token[i])

    row_ids = range(len(documents))

    df = pd.DataFrame({'row_id':row_ids, 'document':documents,'token':tokens, 'label':labels, 'word':words})

    df.to_csv('train_sub.csv', index=False)

    return row_ids, documents, tokens, labels


og_document, og_token, og_label, og_tok_id = reader()
analyzer(og_document=og_document, og_token=og_token, og_label=og_label, og_tok_id=og_tok_id)
