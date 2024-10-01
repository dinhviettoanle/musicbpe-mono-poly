import argparse
import yaml
import os
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import torch
from torchinfo import summary

from MTCFeatures import MTCFeatureLoader
from src.phrase_segmentation_utils import *
import pandas as pd
from src.miditok import bpe
from src.tokenizer import *
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from datetime import datetime
from transformers import BertConfig, BertForTokenClassification
from transformers import GPT2Config, GPT2ForTokenClassification

from pprint import pprint

from src.transformers_trainer import *
from pynvml import *
import time

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

seed_model = 42

PATH_CKPT = "/mnt/gestalt/home/dinhviettoanle/tokenization_models"

FOLDER_MIDI = "corpus/mtc/MTC-FS-INST-2.0/midi_mono"
FEATURE_FILE = 'corpus/essen_phrases/MTC-FS-INST-2.0_sequences-1.1.jsonl.gz'
SAVE_PRECOMPUTED_DATA = 'mtc_clfdata_{}_bpe{}.feather'

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
    
    
def make_phrase_data_monophonic():
    print("[info] Preparing data...")
    seqs = MTCFeatureLoader(FEATURE_FILE).sequences()
    seqs = list(seqs)

    print("N sequences:", len(seqs))
    
    copy_files(seqs)
    
    phrase_data = []
    for ii, x in tqdm(enumerate(seqs), total=len(seqs), desc='Retrieving segmentation info'):
        data = {
            'id': x['id'],
            'filename' : os.path.join(FOLDER_MIDI, f"{x['id']}.mid"),
            'pitch': x['features']['pitch'],
            'phrase_idx': x['features']['phrase_ix'],
            'phrase_end': x['features']['phrase_end']
        }

        phrase_data.append(data)
        
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
        train_data_path = ['corpus', 'essen_phrases', 'midi']
        train_tokenized_data_path = ['data_tokenized', tokenizer_name, 'essen_phrases', 'train']
        train_tokenized_bpe_data_path = ['data_tokenized_bpe', tokenizer_name, 'essen_phrases', 'train']

        train_dirpath = Path(*train_tokenized_data_path)
        if train_dirpath.exists() and train_dirpath.is_dir(): shutil.rmtree(train_dirpath)
        train_bpe_dirpath = Path(*train_tokenized_bpe_data_path)
        if train_bpe_dirpath.exists() and train_bpe_dirpath.is_dir(): shutil.rmtree(train_bpe_dirpath)
        
        tokenizer_bpe = bpe(TokenizerClass, nb_velocities=nb_velocities)
        tokenizer_name = tokenizer_name.__class__.__bases__[0].__name__
        
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


def make_token_phrase_data_one_piece(tokenizer, data):
    midi = MidiFile(data['filename'])
    tokens = tokenizer.midi_to_tokens(midi)[0]
    tokenizer_name = tokenizer.__class__.__bases__[0].__name__


    pointer_score = -1 if 'Absolute' in tokenizer_name else 0
    dict_token_phrase_data = {
        'piece_id': data['id'],
        'tokens': [],
        'token_phrase_idx': [],
        'pitch_phrase': [],
        'frmt_tokens': [],
    }

    for token in tokens:
        frmt_token = tokenizer.vocab._token_to_event[token]
        if 'Pitch' in frmt_token:
            pointer_score = min(pointer_score + 1, len(data['pitch']) - 1)

        dict_token_phrase_data['tokens'].append(token)
        dict_token_phrase_data['token_phrase_idx'].append(data['phrase_idx'][pointer_score])
        dict_token_phrase_data['pitch_phrase'].append(data['pitch'][pointer_score])
        dict_token_phrase_data['frmt_tokens'].append(frmt_token)
    
    return dict_token_phrase_data


def prepare_data_monophonic(tokenizer, tokenizer_name, TokenizerClass, config, precompute_data):
    bpe_merges = config['bpe_merges']
    tokenizer_bpe = train_bpe_tokenizer_for_data(TokenizerClass, config)
    
    if precompute_data:
        phrase_data = make_phrase_data_monophonic()
            
        supertokens_set = make_supertokens_set(tokenizer_bpe)
        
        list_token_phrase = []
        for data in tqdm(phrase_data, desc='Aligning tokens / segmentation info'):
            token_phrase_data = make_token_phrase_data_one_piece(tokenizer, data)
            list_token_phrase.append(token_phrase_data)

        df_token_phrase = pd.DataFrame(list_token_phrase).drop(columns=['pitch_phrase', 'frmt_tokens'])
        df_token_phrase['start_of_phrase_onehot'] = df_token_phrase.token_phrase_idx.apply(make_start_of_phrase_onehot)
        
        df_token_phrase['tokens_bpe'] = df_token_phrase.tokens.parallel_apply(lambda s: tokenizer_bpe.apply_bpe(list(s)))
        df_token_phrase['aligned_tokens_bpe'] = df_token_phrase.tokens_bpe.progress_apply(lambda s: decompose_bpe_align(s, tokenizer_bpe, element_to_insert='length')[1])
        
        df_token_phrase[['start_of_phrase_nonbpe', 'start_of_phrase_bpe']] = df_token_phrase.progress_apply(
            lambda row: compute_start_of_phrase_from_one_hot(row.start_of_phrase_onehot, row.aligned_tokens_bpe, row.tokens_bpe, supertokens_set),
            axis='columns', result_type='expand'
        )
        
        df_token_phrase['length'] = df_token_phrase.tokens.apply(len)
        
        df_token_phrase.to_feather(SAVE_PRECOMPUTED_DATA.format(tokenizer_name, bpe_merges))
    else:
        print("[info] Loading from", SAVE_PRECOMPUTED_DATA.format(tokenizer_name, bpe_merges))
        df_token_phrase = pd.read_feather(SAVE_PRECOMPUTED_DATA.format(tokenizer_name, bpe_merges))
    
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
    config["model_dir"] = os.path.join(
        PATH_CKPT, f"exp231_tf_clfdata/{tokenizer_name}_no_bpe_{config.get('bpe_merges')}_{datetime.now().strftime('%m%d_%H%M')}"
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
    
    df_token_phrase, _ = prepare_data_monophonic(tokenizer, tokenizer_name, TokenizerClass, config, precompute_data)
    
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
    device = torch.device(config['device'])
    
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
        PATH_CKPT, f"exp231_tf_clfdata/{n_merges}/{tokenizer_name}_with_bpe_{datetime.now().strftime('%m%d_%H%M')}"
    )

    print(">>> Initial config <<<")
    pprint(config)
    print()
    
    ## Prepare dataset
    tokenizer = bpe(TokenizerClass, nb_velocities=nb_velocities)
    df_token_phrase, tokenizer_bpe = prepare_data_monophonic(tokenizer, tokenizer_name, TokenizerClass, config, precompute_data)
    
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
    device = torch.device(config['device'])
    
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
    MEMORY_MIN = 4
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
        while True:
            try:
                main(config_file, seed_split=seed_split, config=config)
                break
            except FileNotFoundError as e:
                print(datetime.now().astimezone(), "Not available...")
                time.sleep(60*3)


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
                