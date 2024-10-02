# WARNING: non_bpe MUST be evaluated with chunk_when = before 


import argparse
import yaml
import os
import torch.optim as optim
import torch
from torchinfo import summary
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

from MTCFeatures import MTCFeatureLoader
from src.phrase_segmentation_utils import (
    make_start_of_phrase_onehot,
    decompose_bpe_align,
    compute_start_of_phrase_from_one_hot,
    check_balanced_accuracy,
    PhraseDataset,
    create_subsets,
)
from src.utils import make_supertokens_set, save_config
import pandas as pd
from src.miditok import bpe
from src.tokenizer import *
import torch
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
from transformers import BertConfig, BertForTokenClassification
from transformers import GPT2Config, GPT2ForTokenClassification
import shutil

from pprint import pprint
from src.transformers_trainer import get_lr_scheduler, Trainer
from pynvml import nvmlInit, nvmlDeviceGetMemoryInfo, nvmlDeviceGetHandleByIndex
import time

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

seed_model = 42

PATH_CKPT = "/mnt/nfs_share_magnet2/ldinhvie/tokenization_models"

FOLDER_MIDI = "corpus/mtc_piano_type0"
FEATURE_FILE = 'corpus/essen_phrases/MTC-FS-INST-2.0_sequences-1.1.jsonl.gz'
SAVE_PRECOMPUTED_DATA = 'mtc_piano_clfdata_{}_bpe{}_chunk{}.feather'

# ==================================== DATA ==========================================
def copy_files(seqs):
    dst = f"corpus/essen_phrases/midi"
    if os.path.exists(dst) and os.path.isdir(dst): shutil.rmtree(dst)
    os.makedirs(dst)
    print(f"Copy {len(seqs)} sequences")
    for ii, x in enumerate(seqs):
        # print(ii, x['id'])
        origin = os.path.join(FOLDER_MIDI, f"{x['id']}.mid")
        shutil.copy(origin, dst)
    
    
def make_phrase_data_polyphonic():
    print("[info] Preparing data...")
    seqs = MTCFeatureLoader(FEATURE_FILE).sequences()
    seqs = list(seqs)

    print("N sequences:", len(seqs))
    
    phrase_data = []
    for ii, x in tqdm(enumerate(seqs), total=len(seqs), desc='Retrieving segmentation info'):
        # Remove 6/8 pieces
        # ts = list(set(x['features']['timesignature']))
        # if '6/8' in ts: continue
        
        filename = os.path.join(FOLDER_MIDI, f"{x['id']}_piano_type0.mid")
        if os.path.exists(filename):
            data = {
                'id': x['id'],
                'ii': ii, 
                'filename' : filename,
                'midipitch': x['features']['midipitch'],
                'phrase_idx': x['features']['phrase_ix'],
                # 'phrase_end': x['features']['phrase_end']
            }
            phrase_data.append(data)
    
    print("Total arrangements:", len(phrase_data))
    return phrase_data


def train_bpe_tokenizer_for_data(TokenizerClass, config):
    n_merges = config.get('bpe_merges')
    nb_velocities = config.get('nb_velocities')
    tokenizer_name = TokenizerClass.__name__
    
    ## Train BPE
    if config.get('bpe_savepath'):
        tokenizer_bpe = bpe(TokenizerClass, nb_velocities=nb_velocities)
        tokenizer_name = tokenizer_bpe.__class__.__bases__[0].__name__
        print("[info] Loading from", config['bpe_savepath'].format(n_merges))
        tokenizer_bpe.load_params(config['bpe_savepath'].format(n_merges))
        tokenizer_bpe.save_params(Path(f"{config['model_dir']}/train.bpe"))
    else:
        train_data_path = ['corpus', 'mtc_piano']
        train_tokenized_data_path = ['data_tokenized', tokenizer_name, 'mtc_piano', 'train']
        train_tokenized_bpe_data_path = ['data_tokenized_bpe', tokenizer_name, 'mtc_piano', 'train']

        train_dirpath = Path(*train_tokenized_data_path)
        if train_dirpath.exists() and train_dirpath.is_dir(): shutil.rmtree(train_dirpath)
        train_bpe_dirpath = Path(*train_tokenized_bpe_data_path)
        if train_bpe_dirpath.exists() and train_bpe_dirpath.is_dir(): shutil.rmtree(train_bpe_dirpath)
        
        tokenizer_bpe = bpe(TokenizerClass, nb_velocities=nb_velocities)
        tokenizer_name = tokenizer_bpe.__class__.__bases__[0].__name__
        
        midi_paths = list(Path(*train_data_path).glob('**/*.mid'))
        tokenizer_bpe.tokenize_midi_dataset(midi_paths, Path(*train_tokenized_data_path))

        tokenizer_bpe.bpe(
            tokens_path=Path(*train_tokenized_data_path), 
            n_merges=n_merges,
            out_dir=Path(*train_tokenized_bpe_data_path), 
            files_lim=None
        )
        tokenizer_bpe.save_params(Path(f"{config['model_dir']}/train.bpe"))
    
    vocab_size = len(tokenizer_bpe.vocab)
    print("BPE Vocab size :", vocab_size)
    print("BPE Tokenizer :", tokenizer_name)
    
    return tokenizer_bpe


def make_token_phrase_data_one_piece(tokenizer_full, tokenizer_velocitymute, data, VERBOSE=False, FORCE=False):
    midi_file = MidiFile(data['filename'])
    tokens = tokenizer_full(midi_file)[0]

    mono_midipitch = data['midipitch']
    mono_phrase_idx = data['phrase_idx']
    
    if VERBOSE:
        for p, i in zip(mono_midipitch, mono_phrase_idx):
            print(p, i)
        print("=========================")

    mel_position = []
    # Check that we have the same number of mono pitch and melody tokens
    for ii, token in enumerate(tokens):
        frmt_token = tokenizer_full.vocab._token_to_event[token]
        if 'Velocity_127' in frmt_token:
            mel_position.append(ii-1)

    if not(FORCE):
        assert len(mel_position) == len(mono_midipitch), "Pb LENGTH: as tokens {} vs. as mono {}".format(len(mel_position), len(mono_midipitch))

    # Compute token_phrase_idx
    error = 0
    token_phrase_idx = [None] * len(tokens) 
    pointer_mono = 0
    current_phrase_idx = mono_phrase_idx[pointer_mono]
    for ii, token in enumerate(tokens):
        frmt_token = tokenizer_full.vocab._token_to_event[token]
        if ii in mel_position:
            if VERBOSE: print(ii, frmt_token, mono_midipitch[pointer_mono], current_phrase_idx)
            if not(FORCE):
                assert frmt_token.replace('Pitch_', '') == str(mono_midipitch[pointer_mono]), f"Pb PITCH {ii} : Pitch@{frmt_token} (token) vs. Pitch@{mono_midipitch[pointer_mono]} (mono)"
            else:
                if frmt_token.replace('Pitch_', '') != str(mono_midipitch[pointer_mono]):
                    # We still want to force the destiny, maybe just 1 or 2 notes missing...
                    if VERBOSE: print(f"(FORCE) Pb PITCH {ii} : Pitch@{frmt_token} (token) vs. Pitch@{mono_midipitch[pointer_mono]} (mono)")
                    pointer_mono += 1
                    error += 1
            current_phrase_idx = mono_phrase_idx[pointer_mono]
            pointer_mono += 1
        # print(frmt_token, current_phrase_idx)
        token_phrase_idx[ii] = current_phrase_idx
    
    if FORCE and error > 0: print("(FORCE) {} -- errors: {}".format(data['id'], error))
    if FORCE and error > 10:
        raise AssertionError('(FORCE) Still too many errors {} / {}'.format(error, len(mono_midipitch)))
    
    # Remove velocities
    dict_token_phrase_data = {
        'piece_id': data['id'],
        'transposition': 0,
        'tokens': [],
        'token_phrase_idx': [],
        'frmt_tokens': [],
    }
    
    for ii, (token, phrase_idx) in enumerate(zip(tokens, token_phrase_idx)):
        frmt_token = tokenizer_full.vocab._token_to_event[token]
        if 'Velocity' in frmt_token: continue
        
        velocity_mute_token = tokenizer_velocitymute.vocab.event_to_token[frmt_token]
        dict_token_phrase_data['tokens'].append(velocity_mute_token)
        dict_token_phrase_data['token_phrase_idx'].append(phrase_idx)
        dict_token_phrase_data['frmt_tokens'].append(frmt_token)
        
    return dict_token_phrase_data



def split_into_chunks(df_token_phrase, chunk_size, column_tokens='tokens', columns_to_keep=[]):
    print("Chunking...")
    phrase_data_split_list = []
    column_names = df_token_phrase.columns
    # loop on pieces
    for i_row, phrase_data in df_token_phrase.iterrows():
        tokens = phrase_data[column_tokens]
        for ii, idx_beg in enumerate(range(0, len(tokens), chunk_size)):
            idx_end = min(len(tokens), idx_beg + chunk_size)
            dict_data_split = dict()
            for col_name in column_names:
                if col_name not in columns_to_keep and isinstance(phrase_data[col_name], list) or isinstance(phrase_data[col_name], np.ndarray):
                    dict_data_split[col_name] = phrase_data[col_name][idx_beg:idx_end]
                else:
                    dict_data_split[col_name] = phrase_data[col_name]
                    
            phrase_data_split_list.append(dict_data_split)
            
    return phrase_data_split_list




def prepare_data_polyphonic(tokenizer, tokenizer_name, TokenizerClass, config, precompute_data):
    bpe_merges = config['bpe_merges']
    tokenizer_bpe = train_bpe_tokenizer_for_data(TokenizerClass, config)
    tokenizer_full = bpe(REMI, nb_velocities=127)
    
    if precompute_data:
        phrase_data = make_phrase_data_polyphonic()
            
        supertokens_set = make_supertokens_set(tokenizer_bpe)
        
        list_token_phrase = []
        for data in tqdm(phrase_data, desc='Aligning tokens / segmentation info'):
            try:
                token_phrase_data = make_token_phrase_data_one_piece(tokenizer_full, tokenizer, data, FORCE=True)
                list_token_phrase.append(token_phrase_data)
            except Exception as e:
                print("Error for {} ({}) => {}".format(data['id'], data['ii'], e))
        
        print(f"Passed: {len(list_token_phrase)} / {len(phrase_data)} ({len(list_token_phrase)/len(phrase_data)})")
        
        df_token_phrase = pd.DataFrame(list_token_phrase).drop(columns=['frmt_tokens'])
        df_token_phrase['start_of_phrase_onehot'] = df_token_phrase.token_phrase_idx.apply(make_start_of_phrase_onehot)
        
        if config['chunk_when'] == 'before':
            df_token_phrase = split_into_chunks(df_token_phrase, config['chunk_size'], column_tokens='tokens') # Split into chunks BEFORE BPE
            df_token_phrase = pd.DataFrame(df_token_phrase)
        
        df_token_phrase['tokens_bpe'] = df_token_phrase.tokens.parallel_apply(lambda s: tokenizer_bpe.apply_bpe(list(s)))
        df_token_phrase['aligned_tokens_bpe'] = df_token_phrase.tokens_bpe.progress_apply(lambda s: decompose_bpe_align(s, tokenizer_bpe, element_to_insert='length')[1])
        
        df_token_phrase[['start_of_phrase_nonbpe', 'start_of_phrase_bpe']] = df_token_phrase.progress_apply(
            lambda row: compute_start_of_phrase_from_one_hot(row.start_of_phrase_onehot, row.aligned_tokens_bpe, row.tokens_bpe, supertokens_set),
            axis='columns', result_type='expand'
        )
        
        if config['chunk_when'] == 'after':
            df_token_phrase = split_into_chunks(df_token_phrase, config['chunk_size'], column_tokens='tokens_bpe', columns_to_keep=['tokens', 'start_of_phrase_nonbpe', 'aligned_tokens_bpe']) # Split into chunks AFTER BPE
            df_token_phrase = pd.DataFrame(df_token_phrase)
        
        df_token_phrase['length'] = df_token_phrase.tokens.apply(len)
        df_token_phrase['length_bpe'] = df_token_phrase['tokens_bpe'].apply(len)
        
        df_token_phrase.to_feather(SAVE_PRECOMPUTED_DATA.format(tokenizer_name, bpe_merges, config['chunk_when']))
    else:
        print("[info] Loading from", SAVE_PRECOMPUTED_DATA.format(tokenizer_name, bpe_merges, config['chunk_when']))
        try:
            df_token_phrase = pd.read_feather(SAVE_PRECOMPUTED_DATA.format(tokenizer_name, bpe_merges, config['chunk_when']))
        except FileNotFoundError as e:
            raise FileNotFoundError(e, "Have you run --precompute_data?")
    
    print("Checking NonBPE column")
    check_balanced_accuracy(df_token_phrase, column_start_of_phrase='start_of_phrase_nonbpe')
    print("Checking BPE column")
    check_balanced_accuracy(df_token_phrase, column_start_of_phrase='start_of_phrase_bpe')
    print()
    
    
    return df_token_phrase, tokenizer_bpe


# ======================================================================================================================
# ======================================================================================================================

def train_tf_no_bpe(TokenizerClass, external_config, precompute_data=False):
    tokenizer_name = TokenizerClass.__name__
    nb_velocities = external_config["nb_velocities"]
    
    config = external_config
    config['seed_model'] = seed_model
    config['model_name'] = 'tf_test'
    config['chunk_when'] = 'before'
    config["model_dir"] = os.path.join(
        PATH_CKPT, f"exp232_tf_clfdata/{tokenizer_name}_no_bpe_{config.get('bpe_merges')}_{datetime.now().strftime('%m%d_%H%M')}"
    )

    print(">>> Initial config <<<")
    pprint(config)
    print()
    
    ## Prepare dataset
    tokenizer = bpe(TokenizerClass, nb_velocities=nb_velocities)
    vocab_size = len(tokenizer.vocab)
    config['vocab_size'] = vocab_size
    print("Vocab size :", vocab_size)
    print("Tokenizer :", tokenizer_name)
    
    df_token_phrase, _ = prepare_data_polyphonic(tokenizer, tokenizer_name, TokenizerClass, config, precompute_data)
    
    ## Split train/test
    np.random.seed(config['seed_split'])
    size_test = int(0.2 * len(df_token_phrase))
    test_pieces = np.random.choice(df_token_phrase.piece_id, size_test)
    df_token_phrase['set'] = df_token_phrase.piece_id.apply(lambda piece_id: 'test' if piece_id in test_pieces else 'train')
    print("Test size :", size_test)
    config['test_pieces'] = str(list(test_pieces))
    config['size_test'] = size_test
    
    ## Data loader    
    df_train = df_token_phrase.loc[df_token_phrase.set == 'train'].copy()
    
    
    column_label = 'start_of_phrase_nonbpe'
    check_balanced_accuracy(df_train, column_start_of_phrase=column_label)
    train_dataset = PhraseDataset(
        tokens=df_train['tokens'].tolist(), 
        label=df_train[column_label].tolist(),
        max_len=config['max_position_embeddings'],
        token_pad=tokenizer['PAD_None'],
        tag_pad=2
    )
    
    train_set, val_set = create_subsets(train_dataset, [0.3])

    print("Size training:", len(train_set), "Size val:", len(val_set), "Total:", len(train_set) + len(val_set))
    print("Max token ID:", max(df_train.tokens.apply(max)))
    
    train_dataloader = DataLoader(
        train_set,
        batch_size=config['batch_size'],
        shuffle=True,
    )

    val_dataloader = DataLoader(
        val_set,
        batch_size=config['batch_size'],
        shuffle=True,
    )
    
    output_size = 3
    torch.manual_seed(seed_model)

    if config['model_base'] == 'GPT2':
        model_config = GPT2Config(
            vocab_size=len(tokenizer), 
            hidden_size=config['hidden_size'],
            max_position_embeddings=config['max_position_embeddings'],
            num_hidden_layers=config['num_hidden_layers'], 
            num_attention_heads=config['num_attention_heads'], 
            is_decoder=True,
            output_attentions=True,
            add_cross_attention=False,
            padding_token_id=tokenizer['PAD_None'],
            num_labels=output_size
        )
        model = GPT2ForTokenClassification(model_config)
    
    elif config['model_base'] == 'BERT':
        model_config = BertConfig(
            vocab_size=len(tokenizer), 
            hidden_size=config['hidden_size'],
            max_position_embeddings=config['max_position_embeddings'],
            num_hidden_layers=config['num_hidden_layers'], 
            num_attention_heads=config['num_attention_heads'], 
            is_decoder=False,
            output_attentions=True,
            add_cross_attention=False,
            padding_token_id=tokenizer['PAD_None'],
            num_labels=output_size
        )
        model = BertForTokenClassification(model_config)
        
    
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    lr_scheduler = get_lr_scheduler(optimizer, config["epochs"], verbose=True)
    criterion = None
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    print("Model on", device)
    print(model)
    config['model_architecture'] = str(model)
    config['model_summary'] = str(summary(model))
    
    print(">>> Final config <<<")
    pprint({k:v for k,v in config.items() if k not in ['model_summary', 'model_architecture', 'test_pieces']})
    
    ## Training
    trainer = Trainer(
        model=model,
        epochs=config["epochs"],
        train_dataloader=train_dataloader,
        train_steps=config["train_steps"],
        val_dataloader=val_dataloader,
        val_steps=config["val_steps"],
        task='clf',
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_frequency=config["checkpoint_frequency"],
        lr_scheduler=lr_scheduler,
        early_stopping_patience=config["early_stopping_patience"],
        device=device,
        parallel_devices=config['parallel_devices'],
        model_dir=config["model_dir"],
        model_name=config["model_name"],
    )
    
    save_config(config)
    
    trainer.train()
    
    trainer.save_model()
    trainer.save_loss()

# ======================================================================================================================

def train_tf_with_bpe(TokenizerClass, external_config, precompute_data=False):
    tokenizer_name = TokenizerClass.__name__
    n_merges = external_config["bpe_merges"]
    nb_velocities = external_config["nb_velocities"]
    
    config = external_config
    config['seed_model'] = seed_model
    config['model_name'] = 'tf_test'
    config["model_dir"] = os.path.join(
        PATH_CKPT, f"exp232_tf_clfdata/{n_merges}/{tokenizer_name}_with_bpe_{datetime.now().strftime('%m%d_%H%M')}"
    )

    print(">>> Initial config <<<")
    pprint(config)
    print()
    
    ## Prepare dataset
    tokenizer = bpe(TokenizerClass, nb_velocities=nb_velocities)
    df_token_phrase, tokenizer_bpe = prepare_data_polyphonic(tokenizer, tokenizer_name, TokenizerClass, config, precompute_data)
    
    config['vocab_size'] = len(tokenizer_bpe.vocab)
    df_token_phrase_bpe = df_token_phrase.copy()

    ## Split train/test
    np.random.seed(config['seed_split'])
    size_test = int(0.2 * len(df_token_phrase_bpe))
    test_pieces = np.random.choice(df_token_phrase.piece_id, size_test)
    df_token_phrase_bpe['set'] = df_token_phrase_bpe.piece_id.apply(lambda piece_id: 'test' if piece_id in test_pieces else 'train')
    print("Test size :", size_test)
    config['test_pieces'] = str(list(test_pieces))
    config['size_test'] = size_test
    
    ## Data loader    
    df_train = df_token_phrase_bpe.loc[df_token_phrase_bpe.set == 'train'].copy()
    
    
    column_label = 'start_of_phrase_bpe'
    check_balanced_accuracy(df_train, column_start_of_phrase=column_label)
    train_dataset = PhraseDataset(
        tokens=df_train['tokens_bpe'].tolist(), 
        label=df_train[column_label].tolist(),
        max_len=config['max_position_embeddings'],
        token_pad=tokenizer_bpe['PAD_None'],
        tag_pad=2
    )
    
    train_set, val_set = create_subsets(train_dataset, [0.3])

    print("Size training:", len(train_set), "Size val:", len(val_set), "Total:", len(train_set) + len(val_set))
    print("Max token ID:", max(df_train.tokens.apply(max)))
    
    train_dataloader = DataLoader(
        train_set,
        batch_size=config['batch_size'],
        shuffle=True,
    )

    val_dataloader = DataLoader(
        val_set,
        batch_size=config['batch_size'],
        shuffle=True,
    )
    
    output_size = 3
    torch.manual_seed(seed_model)

    if config['model_base'] == 'GPT2':
        model_config = GPT2Config(
            vocab_size=len(tokenizer_bpe), 
            hidden_size=config['hidden_size'],
            max_position_embeddings=config['max_position_embeddings'],
            num_hidden_layers=config['num_hidden_layers'], 
            num_attention_heads=config['num_attention_heads'], 
            is_decoder=True,
            output_attentions=True,
            add_cross_attention=False,
            padding_token_id=tokenizer_bpe['PAD_None'],
            num_labels=output_size
        )
        model_bpe = GPT2ForTokenClassification(model_config)
    
    elif config['model_base'] == 'BERT':
        model_config = BertConfig(
            vocab_size=len(tokenizer_bpe), 
            hidden_size=config['hidden_size'],
            max_position_embeddings=config['max_position_embeddings'],
            num_hidden_layers=config['num_hidden_layers'], 
            num_attention_heads=config['num_attention_heads'], 
            is_decoder=False,
            output_attentions=True,
            add_cross_attention=False,
            padding_token_id=tokenizer_bpe['PAD_None'],
            num_labels=output_size
        )
        model_bpe = BertForTokenClassification(model_config)
        
    
    optimizer_bpe = optim.Adam(model_bpe.parameters(), lr=config["lr"])
    lr_scheduler = get_lr_scheduler(optimizer_bpe, config["epochs"], verbose=True)
    criterion = None
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    
    print("Model on", device)
    print(model_bpe)
    config['model_architecture'] = str(model_bpe)
    config['model_summary'] = str(summary(model_bpe))
    
    print(">>> Final config <<<")
    pprint({k:v for k,v in config.items() if k not in ['model_summary', 'model_architecture', 'test_pieces']})
    
    ## Training
    trainer_bpe = Trainer(
        model=model_bpe,
        epochs=config["epochs"],
        train_dataloader=train_dataloader,
        train_steps=config["train_steps"],
        val_dataloader=val_dataloader,
        val_steps=config["val_steps"],
        task='clf',
        criterion=criterion,
        optimizer=optimizer_bpe,
        checkpoint_frequency=config["checkpoint_frequency"],
        lr_scheduler=lr_scheduler,
        early_stopping_patience=config["early_stopping_patience"],
        device=device,
        parallel_devices=config['parallel_devices'],
        model_dir=config["model_dir"],
        model_name=config["model_name"],
    )
    
    save_config(config)
    
    trainer_bpe.train()
    
    trainer_bpe.save_model()
    trainer_bpe.save_loss()


# ======================================================================================================================
# ======================================================================================================================


def main(config_file, seed_split=None, precompute_data=False, config=None):
    if config is None:
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
    TokenizerClass = globals()[config['TokenizerClass']]
    config['seed_split'] = seed_split
    
    if not(config['bpe']):
        train_tf_no_bpe(TokenizerClass, config, precompute_data)
    else:
        train_tf_with_bpe(TokenizerClass, config, precompute_data)


def loop(config_file, begin_seed_split):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
            
    nvmlInit()
    MEMORY_MIN = 2
    device_number = int(config['device'].replace('cuda:', ''))
    while True:
        memory_free = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(device_number)).free / 1e9
        print(datetime.now().astimezone(), memory_free)
        if memory_free > MEMORY_MIN:
            print("Some free memory on cuda:{} -> {} !".format(device_number, memory_free))
            time.sleep(20)
            break
        time.sleep(10)
    
    for seed_split in range(begin_seed_split, 3):
        print("================================== BEGIN FOR SEED SPLIT = {} =========================".format(seed_split))
        main(config_file, seed_split=seed_split, config=config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed_split", type=int, default=0)
    parser.add_argument('--precompute_data', action='store_true')
    parser.add_argument('--loop', action='store_true')
    args = parser.parse_args()
    
    if args.loop:
        loop(args.config, args.seed_split)
    else:
        main(args.config, args.seed_split, args.precompute_data)