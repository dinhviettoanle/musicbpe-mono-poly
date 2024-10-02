import argparse
from src.miditok import bpe
from src.tokenizer import *
import torch

TokenizerClass = REMIVelocityMute
nb_velocities = 1
tokenizer_name = TokenizerClass.__name__


def train_mono(tokenizer_init_path, n_merges, output_file, bypass_tokenize_init=False):
    tokenizer_name = 'REMIVelocityMute'
    train_data_path = ['corpus', 'essen_phrases', 'midi']
    train_tokenized_data_path = ['data_tokenized', tokenizer_name, 'essen_phrases', 'train']
    train_tokenized_bpe_data_path = ['data_tokenized_bpe', tokenizer_name, 'essen_phrases', 'train']

    if not(bypass_tokenize_init):
        train_dirpath = Path(*train_tokenized_data_path)
        if train_dirpath.exists() and train_dirpath.is_dir(): shutil.rmtree(train_dirpath)
        train_bpe_dirpath = Path(*train_tokenized_bpe_data_path)
        if train_bpe_dirpath.exists() and train_bpe_dirpath.is_dir(): shutil.rmtree(train_bpe_dirpath)

    tokenizer_bpe = bpe(TokenizerClass, nb_velocities=nb_velocities)
    tokenizer_name = tokenizer_name.__class__.__bases__[0].__name__

    if tokenizer_init_path is not None:
        print("Loading from", tokenizer_init_path)
        tokenizer_bpe.load_params(tokenizer_init_path)
        print("Vocab loaded", tokenizer_bpe.vocab)
    
    if not(bypass_tokenize_init):
        midi_paths = list(Path(*train_data_path).glob('**/*.mid'))
        tokenizer_bpe.tokenize_midi_dataset(midi_paths, Path(*train_tokenized_data_path))
    
    print("Init vocab:", tokenizer_bpe.vocab)
    
    tokenizer_bpe.bpe(
        tokens_path=Path(*train_tokenized_data_path), 
        n_merges=n_merges,
        out_dir=Path(*train_tokenized_bpe_data_path), 
        files_lim=None
    )
    
    print("Final vocab:", tokenizer_bpe.vocab)
    
    tokenizer_bpe.save_params(output_file)
    torch.save(tokenizer_bpe.supertoken_frq, f"{output_file}.frq")
    
    
    
def train_poly(tokenizer_init_path, n_merges, output_file, bypass_tokenize_init=False):
    tokenizer_name = 'REMIVelocityMute'
    train_data_path = ['corpus', 'mtc_piano_type0']
    train_tokenized_data_path = ['data_tokenized', tokenizer_name, 'mtc_piano', 'train']
    train_tokenized_bpe_data_path = ['data_tokenized_bpe', tokenizer_name, 'mtc_piano', 'train']

    if not(bypass_tokenize_init):
        train_dirpath = Path(*train_tokenized_data_path)
        if train_dirpath.exists() and train_dirpath.is_dir(): shutil.rmtree(train_dirpath)
        train_bpe_dirpath = Path(*train_tokenized_bpe_data_path)
        if train_bpe_dirpath.exists() and train_bpe_dirpath.is_dir(): shutil.rmtree(train_bpe_dirpath)

    tokenizer_bpe = bpe(TokenizerClass, nb_velocities=nb_velocities)
    tokenizer_name = tokenizer_name.__class__.__bases__[0].__name__

    if tokenizer_init_path is not None:
        print("Loading from", tokenizer_init_path)
        tokenizer_bpe.load_params(tokenizer_init_path)
        print("Vocab loaded", tokenizer_bpe.vocab)
    
    if not(bypass_tokenize_init):
        midi_paths = list(Path(*train_data_path).glob('**/*.mid'))
        tokenizer_bpe.tokenize_midi_dataset(midi_paths, Path(*train_tokenized_data_path))
    
    print("Init vocab:", tokenizer_bpe.vocab)
    
    tokenizer_bpe.bpe(
        tokens_path=Path(*train_tokenized_data_path), 
        n_merges=n_merges,
        out_dir=Path(*train_tokenized_bpe_data_path), 
        files_lim=None
    )
    
    print("Final vocab:", tokenizer_bpe.vocab)
    
    tokenizer_bpe.save_params(output_file)
    torch.save(tokenizer_bpe.supertoken_frq, f"{output_file}.frq")
    
    


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_init", default=None)
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument('--n_merges', type=int, required=True)
    parser.add_argument("--output_file", type=str, default='train.bpe')
    parser.add_argument("--bypass_tokenize", action='store_true')
    args = parser.parse_args()

    if args.corpus == 'mono':
        train_mono(args.tokenizer_init, args.n_merges, args.output_file, args.bypass_tokenize)
    elif args.corpus == 'poly':
        train_poly(args.tokenizer_init, args.n_merges, args.output_file, args.bypass_tokenize)
        
