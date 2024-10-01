from src.miditok import bpe
from pathlib import Path
import numpy as np
from tqdm import tqdm
import shutil
import os, sys
import pprint
pp = pprint.PrettyPrinter(width=20)

import sklearn.metrics as metrics
import torch
from torchinfo import summary

from src.utils import *
from src.tokenizer import *

from torch.utils.data import DataLoader

from src.phrase_segmentation_utils import *
from exp232_clfdata_tf import SAVE_PRECOMPUTED_DATA
from sklearn.dummy import DummyClassifier


tqdm.pandas()
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
np.random.seed(42)

ONLY_TEST = True

ckpt_path = os.path.join(sys.argv[1])
print(ckpt_path)

device = sys.argv[2] if len(sys.argv) > 1 else 'cuda:0'


config = read_config(ckpt_path)
bpe_merges = config['bpe_merges']

chunk_when = config['chunk_when']
if not(config['bpe']):
    chunk_when = 'before'


TokenizerClass = globals()[config['TokenizerClass']]

tokenizer = bpe(TokenizerClass, nb_velocities=config.get('nb_velocities', 1))
tokenizer_bpe = bpe(TokenizerClass, nb_velocities=config.get('nb_velocities', 1))
tokenizer_name = tokenizer_bpe.__class__.__bases__[0].__name__


print("[info] Loading from", SAVE_PRECOMPUTED_DATA.format(tokenizer_name, bpe_merges, chunk_when))
df_token_phrase = pd.read_feather(SAVE_PRECOMPUTED_DATA.format(tokenizer_name, bpe_merges, chunk_when))

model = torch.load(os.path.join(ckpt_path, 'best_loss.pt'), map_location=torch.device('cpu'))

device = torch.device(device if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

np.random.seed(config['seed_split'])
size_test = int(0.2 * len(df_token_phrase))
test_pieces = np.random.choice(df_token_phrase.piece_id, size_test)
print("Test size", size_test)

assert eval(config['test_pieces']) == list(test_pieces)


if ONLY_TEST:
    df_token_phrase = df_token_phrase.loc[df_token_phrase['piece_id'].isin(test_pieces)]
    df_token_phrase['set'] = ['test'] * len(df_token_phrase)
else:
    df_token_phrase['set'] = df_token_phrase.piece_id.progress_apply(lambda piece_id: 'test' if piece_id in test_pieces else 'train')
    
# ===============================


column_label = 'start_of_phrase_bpe' if config['bpe'] else 'start_of_phrase_nonbpe'
token_label = 'tokens_bpe' if config['bpe'] else 'tokens'
dataset = PhraseDataset(
    tokens=df_token_phrase[token_label].tolist(), 
    label=df_token_phrase[column_label].tolist(),
    max_len=config['max_position_embeddings'],
    token_pad=tokenizer['PAD_None'],
    tag_pad=2
)

print("Size test:", len(dataset))

test_dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    # collate_fn=pad_collate
)

model.eval()

predictions = []
true_labels = []
all_preds, all_true_labels = [], []

# dummy_clf = DummyClassifier(strategy='uniform')
# X = []
# y = []
# for sample in tqdm(dataset, desc='making dset'):
#     for tok, label in zip(sample['ids'], sample['target']):
#         X.append(tok)
#         y.append(label)
#         if label == 2: break
# X = np.array(X)
# y = np.array(y)
# dummy_clf.fit(X, y)


with torch.no_grad():
    for index, dataset in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        batch_input_ids = dataset['ids'].to(device, dtype = torch.long)
        batch_att_mask = dataset['att_mask'].to(device, dtype = torch.long)
        batch_target = dataset['target'].to(device, dtype = torch.long)
        

        output = model(batch_input_ids, 
                    token_type_ids=None,
                    attention_mask=batch_att_mask,
                    labels=batch_target)

        step_loss = output.loss
        eval_prediction = output.logits

        
        # accuracy
        # eval_prediction = dummy_clf.predict(batch_input_ids.flatten())
        # eval_prediction = np.expand_dims(eval_prediction, axis=0)
        
        eval_prediction = np.argmax(eval_prediction.detach().to('cpu').numpy(), axis = 2)
        actual = batch_target.to('cpu').numpy()
        
        predictions.extend(eval_prediction.flatten())
        true_labels.extend(actual.flatten())
        
        # Retrieve the correct sequence
        lengths = batch_att_mask.sum(axis=1)
        preds, targets = [], []
        for b, length in enumerate(lengths):
            preds.append(eval_prediction[b][:length])
            targets.append(actual[b][:length])
        eval_prediction = np.concatenate(preds)
        actual = np.concatenate(targets)
        all_preds.extend(preds)
        all_true_labels.extend(targets)

# ==================================

dict_perfo = {
    'test_acc': metrics.accuracy_score(y_true=true_labels, y_pred=predictions),
    'test_recall': metrics.recall_score(y_true=true_labels, y_pred=predictions, average='macro', labels=[0, 1]),
    'test_precision': metrics.precision_score(y_true=true_labels, y_pred=predictions, average='macro', labels=[0, 1]),
    'test_macrof1': metrics.f1_score(y_true=true_labels, y_pred=predictions, average='macro', labels=[0, 1]),
    'test_weightedf1': metrics.f1_score(y_true=true_labels, y_pred=predictions, average='weighted', labels=[0, 1]),
    'class1_f1': metrics.f1_score(y_true=true_labels, y_pred=predictions, average='macro', labels=[1]),
    'class1_precision': metrics.precision_score(y_true=true_labels, y_pred=predictions, average='macro', labels=[1]),
    'class1_recall': metrics.recall_score(y_true=true_labels, y_pred=predictions, average='macro', labels=[1]),
}

print()
print(metrics.classification_report(y_true=true_labels, y_pred=predictions, labels=[0, 1]))

pprint.pprint(dict_perfo)


out_perfo_path = os.path.join(ckpt_path, 'perfo.json')
with open(out_perfo_path, 'w') as fp:
    json.dump(dict_perfo, fp, indent=2)
    
    
bpe_or_not = 'bpe' if config['bpe'] else 'nobpe'
after_or_not = ''
if config['bpe']:
    after_or_not = f"_{config['chunk_when']}"
    
torch.save({
    'ckpt': ckpt_path,
    'chunk_when': config['chunk_when'],
    'n_merges': config['bpe_merges'],
    'seed_split': config['seed_split'],
    'preds': all_preds,
    'targets': all_true_labels
}, f"verif_232clf_{bpe_or_not}{after_or_not}.pt")



