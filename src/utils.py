"""
General utils functions
"""
from importlib.metadata import version
from tqdm.auto import tqdm
from pathlib import Path
import os
import shutil
import json
import torch
from miditoolkit import MidiFile
import pandas as pd
import numpy as np
import torch
import multiprocessing as mp
from functools import partial
import time
from concurrent.futures import ProcessPoolExecutor
import rich
from rich import progress

from .miditok import REMI, Octuple, Structured, CPWord, MIDILike


NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
OCTAVES = list(range(11))
NOTES_IN_OCTAVE = len(NOTES)
MIDI_MAX = 100

DEFAULT_SEED = 42

def number_to_note(number: int) -> str:
    """Converts a MIDI number to its string representation "NoteOctave"

    Parameters
    ----------
    number : int
        MIDI number

    Returns
    -------
    str
        String representation of the note
    """
    octave = number // NOTES_IN_OCTAVE
    assert octave in OCTAVES, errors['notes']
    assert 0 <= number <= 127, errors['notes']
    note = NOTES[number % NOTES_IN_OCTAVE]
    return f'{note}{octave}'


def token_id_to_note(tokenizer, token_id):
    """Converts a Pitch token to a readable note representation "NoteOctave"

    Parameters
    ----------
    tokenizer : MIDITokenizer
        Tokenizer with its vocab
    token_id : int
        Token ID

    Returns
    -------
    str
        String representation of the pitch token
    """
    str_event = tokenizer.vocab._token_to_event[token_id]
    if not('_') in str_event:
        return "P"
    midi_pitch = str_event.split('_')[1]
    assert midi_pitch.isdigit()
    return number_to_note(int(midi_pitch))


def midi_to_token(tokenizer, midi_number):
    """Converts a MIDI number into a Pitch token

    Parameters
    ----------
    tokenizer : MIDITokenizer
        Tokenizer with its vocab
    midi_number : int
        Pitch in MIDI format

    Returns
    -------
    int
        Token ID of the corresponding pitch
    """
    for token in tokenizer.vocab._token_to_event:
        frmt_token = tokenizer.vocab._token_to_event[token]
        if tokenizer.vocab.token_type(token) == 'Pitch' and int(frmt_token.split('_')[1]) == midi_number:
            return token
        
    # print_vocab(tokenizer)
    raise IndexError(f"Not found token for MIDI number = {midi_number}")


def query_token(tokenizer, query_type, query_value):
    for token in tokenizer.vocab._token_to_event:
        frmt_token = tokenizer.vocab._token_to_event[token]
        if tokenizer.vocab.token_type(token) == query_type and int(frmt_token.split('_')[1]) == query_value:
            return token
    raise IndexError(f"Not found token for query_type = {query_type} and query_value = {query_value}")


def token_to_midi(tokenizer, token):
    """Converts a Pitch token to its MIDI number

    Parameters
    ----------
    tokenizer : MIDITokenizer
        Tokenizer with its vocab
    token : int
        Token ID

    Returns
    -------
    int
        MIDI number
    """
    str_event = tokenizer.vocab._token_to_event[token]
    midi_pitch = str_event.split('_')[1]
    assert midi_pitch.isdigit()
    return int(midi_pitch)


def show_vocabulary_notes(tokenizer):
    """Show the vocabulary of a tokenizer with readable notes

    Parameters
    ----------
    tokenizer : MIDITokenizer
        Tokenizer

    Returns
    -------
    List[str]
        Vocabulary with readable pitches
    """
    list_vocabulary = []
    for token in tokenizer.vocab._token_to_event:
        frmt_token = tokenizer.vocab._token_to_event[token]
        if tokenizer.vocab.token_type(token) == 'BPE':
            root_events = list(map(int, frmt_token.split('.')[1].split('-')))
            frmt_root_events = list(map(lambda x: tokenizer.vocab._token_to_event[x], root_events))
            
            # # # Only pitches
            list_pitches = [x for x in root_events if tokenizer.vocab.token_type(x) == 'Pitch']
            if len(list_pitches) > 1:
                interval_list = [list_pitches[i+1] - list_pitches[i] for i in range(len(list_pitches) - 1)]
                print(list(map(number_to_note, list_pitches)), interval_list)
                list_vocabulary.append(list(map(lambda x: token_id_to_note(tokenizer, x), list_pitches)))
               
                
            # # # All events
            # list_pitches = [x for x in root_events]
            # if len(list_pitches) > 1:
            #     interval_list = [list_pitches[i+1] - list_pitches[i] for i in range(len(list_pitches) - 1)]
            #     list_vocabulary.append(list_pitches)
    return list_vocabulary


def convert_bpe_to_sequence_frmt(tokenizer, bpe_str, AS_MUSIC=False):
    """Format a BPE supertoken into its list of atomic elements

    Parameters
    ----------
    tokenizer : MIDITokenizer
        BPE-trained tokenizer
    bpe_str : str
        ID of the supertoken

    Returns
    -------
    List[str]
        List of readable atomic elements composing the supertoken
    """
    root_events = list(map(int, bpe_str.split('.')[1].split('-')))
    frmt_root_events = []
    for x in root_events:
        frmt_token = tokenizer.vocab._token_to_event[x]
        if 'Pitch' in frmt_token and AS_MUSIC:
            pitchoctave = number_to_note(int(frmt_token.split('_')[1]))
            frmt_root_events.append(f'Pitch_{pitchoctave}')
        else:
            frmt_root_events.append(frmt_token)
    return frmt_root_events


def print_tokens(tokenizer, tokens, RETURN_LINE=True): 
    """Prints readable token sequence

    Parameters
    ----------
    tokenizer : MIDITokenizer
        Tokenizer (BPE or not)
    tokens : List[int]
        Token sequence
    RETURN_LINE : bool, optional
        Return line at each new token or not, by default True
    """
    end = '\n' if RETURN_LINE else ', ' 
      
    if isinstance(tokenizer, CPWord) or isinstance(tokenizer, Octuple):
        for token in tokens:
            list_elts = []
            for ii, elt in enumerate(token):
                list_elts.append(tokenizer.vocab[ii]._token_to_event[elt])
            print(list_elts)
    
    else:
        for token in tokens:
            frmt_token = tokenizer.vocab._token_to_event[token]
            if tokenizer.vocab.token_type(token) == 'BPE':
                frmt_root_events = convert_bpe_to_sequence_frmt(tokenizer, frmt_token)
                print(f"<BPE {frmt_root_events}>", end=end) 
            elif tokenizer.vocab.token_type(token) == 'Note-On':
                print(">>>>>>>>>>> ", frmt_token, end=end)
            else:
                print(frmt_token, end=end)
                
    if not(RETURN_LINE): print()


def print_vocab(tokenizer, return_bool=False, AS_MUSIC=False):
    """Prints the vocabulary of a tokenizer

    Parameters
    ----------
    tokenizer : MIDITokenizer
        Tokenizer (BPE or not)
    return_bool : bool, optional
        Returns the vocabulary as list or not, by default False

    Returns
    -------
    None | List[str]
        Vocabulary as list of strings
    """
    return_list = []
    for token in tokenizer.vocab._token_to_event:
        frmt_token = tokenizer.vocab._token_to_event[token]
        if tokenizer.vocab.token_type(token) == 'BPE':
            frmt_root_events = convert_bpe_to_sequence_frmt(tokenizer, frmt_token, AS_MUSIC=AS_MUSIC)
            print(token, ":", f"<BPE {frmt_root_events}>")
            return_list.append(f"<BPE {frmt_root_events}>") 
        else:
            print(token, ":", frmt_token)
            return_list.append(frmt_token)
        
    if return_bool:
        return return_list
        

def make_supertokens_set(tokenizer):
    supertokens_set = set()
    for token in tokenizer.vocab._token_to_event:
        if tokenizer.vocab.token_type(token) == 'BPE':
            supertokens_set.add(token)
    return supertokens_set

# ====================================================================================
# ==================== DATASET PREPARATION ===========================================
# ====================================================================================

def count_tokens_corpus(list_genres, tokenizer, folder_corpus='folk_norm', force_skip_fn=None, extension=".mid", VERBOSE=True):
    """Count tokens in a corpus for each genre

    Parameters
    ----------
    list_genres : List[str]
        List of genres
    tokenizer_returner : Callable[None, MIDITokenizer]
        Function that returns the tokenizer
    folder_corpus : str, optional
        Name of the corpus folder (as subfolder of corpus/), by default 'folk_norm'
    force_skip_fn : Callable[[MIDITokenizer, List[int]], bool], optional
        Function to skip a file or not, by default None
    extension : str, optional
        File extension in the folder, by default .mid
    VERBOSE : bool, optional
        Show output when skipped, by default True

    Returns
    -------
    pandas.DataFrame
        DataFrame containing corpus data with columns [genre, filename, tokens_list, count]
    """
    dict_count = {
        'genre': [],
        'file': [],
        'tokens': [],
        'count': [],
    }

    for current_genre in tqdm(list_genres):
        this_genre_count = 0
        current_genre_midi_path = ['corpus', folder_corpus, current_genre]
        files_list = list(Path(*current_genre_midi_path).glob(f'**/*{extension}'))
        for midi_file in tqdm(files_list, leave=False, desc=current_genre, position=1):
            # print(midi_file)
            midi = MidiFile(midi_file)
            tokens = tokenizer.midi_to_tokens(midi)
            
            # Verification que y a qu'un seul track
            if len(tokens) > 1:
                if VERBOSE:
                    print(f"[len(tokens) > 1] Skipping : {midi_file}")
            if len(tokens) == 0:
                continue
            tokens = tokens[0]
                
            # Verification du force_skip_fn
            if force_skip_fn and force_skip_fn(tokenizer, tokens):
                if VERBOSE:
                    print(f"[force_skip_fn] Skipping : {midi_file}")
                continue
            

            dict_count['genre'].append(current_genre)
            dict_count['file'].append(midi_file)
            dict_count['tokens'].append(tokens)
            dict_count['count'].append(len(tokens))

    df_genre_count = pd.DataFrame(dict_count)
    if VERBOSE:
        df_genre_count.groupby('genre').sum().sort_values('count').plot.bar()
        print(df_genre_count.groupby('genre').sum().sort_values('count'))
    
    return df_genre_count



def make_reduced_dataset(df_genre_count, list_considered_genres, lim, COPY_FILES=True, seed=DEFAULT_SEED, VERBOSE=True):
    """Reduces the dataset to make it balanced

    Parameters
    ----------
    df_genre_count : pandas.DataFrame
        DataFrame containing corpus data (computed by count_tokens_corpus)
    list_considered_genres : List[str]
        List of considered genres
    lim : int
        Limit of tokens in each genre
    COPY_FILES : bool, optional
        Copy files in a new folder, by default True
    seed : int, optional
        Seed to randomly chose pieces, by default DEFAULT_SEED
    VERBOSE : bool, optional
        Show the count for each genre, by default True

    Returns
    -------
    pandas.DataFrame
        Subset of df_genre_count, where each genre is balanced
    """
    list_df_reduced = []
    for current_genre in list_considered_genres: 
        df_genre = df_genre_count.loc[df_genre_count.genre == current_genre]

        for n_sampled in range(5000):
            df_reduced = df_genre.sample(random_state=seed, n=min(n_sampled, len(df_genre)), replace=False)
            if df_reduced['count'].sum() >= lim or n_sampled == len(df_genre):
                if VERBOSE:
                    print(f"{current_genre:<15} {n_sampled}/{len(df_genre):<5}", df_reduced['count'].sum())
                list_df_reduced.append(df_reduced)
                break
            
    return pd.concat(list_df_reduced)




def split_train_test_by_genre(df_genre_count, list_considered_genres, lim_test_set, seed=DEFAULT_SEED, VERBOSE=True):
    """Split dataset into train and test sets

    Parameters
    ----------
    df_genre_count : pandas.DataFrame
        DataFrame containing corpus data 
        (computed by count_tokens_corpus for the full corpus, or by make_reduced_dataset for the balanced dataset)
    list_considered_genres : List[str]
        List of considered genres
    lim_test_set : float
        Ratio of the size of the test set
    seed : int, optional
        Seed to randomly populate the train set, by default DEFAULT_SEED
    VERBOSE : bool, optional
        Show the output, by default True

    Returns
    -------
    pandas.DataFrame
        df_genre_count, with train and test set indications
    """
    list_df_sets = []
    for current_genre in list_considered_genres: 
        df_genre = df_genre_count.loc[df_genre_count.genre == current_genre].copy()

        for n_sampled in range(5000):
            df_split = df_genre.sample(random_state=seed, n=min(n_sampled, len(df_genre)), replace=False)
            if df_split['count'].sum() >= lim_test_set:
                if VERBOSE:
                    print(f"{current_genre:<15} TRAIN : {len(df_genre) - n_sampled}/{len(df_genre):<5} TEST : {n_sampled}/{len(df_genre):<5}", df_split['count'].sum())

                df_train = df_genre.loc[~df_genre.index.isin(df_split.index)].copy()
                df_test = df_genre.loc[df_genre.index.isin(df_split.index)].copy()
                df_train['set'] = ['train'] * len(df_train)
                df_test['set'] = ['test'] * len(df_test)

                list_df_sets.append(df_train)
                list_df_sets.append(df_test)
                break

    return pd.concat(list_df_sets)



def tokens_to_bow(tokenizer, tokens):
    """Converts a token sequence into a bag-of-tokens

    Parameters
    ----------
    tokenizer : MIDITokenizer
        Tokenizer (BPE or not)
    tokens : List[int]
        Sequence of tokens

    Returns
    -------
    List[int]
        Histogram of the token counts - each index is the index within the vocabulary
    """
    counts, _ = np.histogram(tokens, bins=len(tokenizer.vocab), range=(0, len(tokenizer.vocab)))
    # Rq : le bins va de 0 à len+1 (donc le dernier compte pour rien)
    return list(counts)


def tokenize_file(tokenizer, file):
    """Tokenizes a file

    Parameters
    ----------
    tokenizer : MIDITokenizer
        Tokenizer (BPE or not)
    file : str || Path
        Path to the MIDI file

    Returns
    -------
    List[int]
        Tokenized piece
    """
    midi = MidiFile(file)
    tokens = tokenizer.midi_to_tokens(midi)[0]
    return tokens


def split_sequences(seq, split_frq):
    """Split a sequence into n-grams

    Parameters
    ----------
    seq : List[int || str]
        List of elements
    split_frq : int
        n in "n-gram"

    Returns
    -------
    List[str]
        Original sequence grouped where each element is grouped into n-gram
    """
    composite_seq = [seq[x:x+split_frq] for x in range(0, len(seq), split_frq)]
    if len(composite_seq[-1]) != split_frq:
        composite_seq.pop()
    
    word_seq = ["-".join(map(str, s)) for s in composite_seq]
    return word_seq


def seq_regular_split_to_bow(seq, vocabulary):
    """Makes a bag-of-tokens from a regular split

    Parameters
    ----------
    seq : List[str]
        Sequence of n-grams
    vocabulary : List[str]
        Vocabulary of the n-grams

    Returns
    -------
    List[int]
        Bag-of-n-grams representing this sequence
    """
    hist = dict([(word, 0) for word in vocabulary])
    for word in seq:
        hist[word] += 1
    return list(hist.values())



# ====================================================================================
# ==================== PYTORCH UTILS =================================================
# ====================================================================================


def save_config(config, filename="config.json"):
    """Saves configuration of an experiment

    Parameters
    ----------
    config : dict
        Dictionary containing configuration of the experiment
    """
    out_config_path = os.path.join(config['model_dir'], filename)
    with open(out_config_path, 'w') as fp:
        json.dump(config, fp, indent=2)
        
        
def read_config(ckpt_dir, filename="config.json"):
    """Saves configuration of an experiment

    Parameters
    ----------
    config : dict
        Dictionary containing configuration of the experiment
    """
    out_config_path = os.path.join(ckpt_dir, filename)
    with open(out_config_path) as fp:
        config = json.load(fp)
    return config
        

def get_pytorch_model_size(model):
    """Prints Pytorch model size in MB

    Parameters
    ----------
    model : torch.nn.Module
        Model
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))
    
    
    
def apply_func_on_dataframe(df, func, progress_bar):
    return df.apply(partial(func, progress_bar=progress_bar), axis=1)


# def parallelize_dataframe(df, func):
#     num_processes = mp.cpu_count()
#     df_split = np.array_split(df, num_processes)
#     with mp.Pool(num_processes) as p:
#         df = pd.concat(p.map(func, df_split))
#     return df


# def parallelize_dataframe(df, func):
#     num_processes = mp.cpu_count()
#     df_split = np.array_split(df, num_processes)

#     with mp.Pool(num_processes) as p:
#         results = []
#         for result in tqdm(p.imap(func, df_split), total=num_processes):
#             results.append(result)

#     df = pd.concat(results)
#     return df


def rich_func(subdf, progress, func, task_id):
    len_of_task = len(subdf)
    result = []
    for n in range(0, len_of_task):
        result.append(func(subdf.iloc[n]))
        progress[task_id] = {"progress": n + 1, "total": len_of_task}
    return result


def parallelize_dataframe(df, func):

    num_processes = mp.cpu_count()
    df_split = np.array_split(df, num_processes)
    
    with rich.progress.Progress(
        "[progress.description]{task.description}",
        rich.progress.BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        rich.progress.TimeRemainingColumn(),
        rich.progress.TimeElapsedColumn(),
        refresh_per_second=100,  # bit slower updates
    ) as progress:
        futures = []  # keep track of the jobs
        with mp.Manager() as manager:
            # this is the key - we share some state between our 
            # main process and our worker functions
            _progress = manager.dict()
            overall_progress_task = progress.add_task("[green]All jobs progress:")

            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                for n, subdf in enumerate(df_split):  # iterate over the jobs we need to run
                    # set visible false so we don't have a lot of bars all at once:
                    task_id = progress.add_task(f"task {n}", visible=False)
                    futures.append(executor.submit(rich_func, subdf, _progress, func, task_id))

                # monitor the progress:
                while (n_finished := sum([future.done() for future in futures])) < len(
                    futures
                ):
                    progress.update(
                        overall_progress_task, completed=n_finished, total=len(futures)
                    )
                    for task_id, update_data in _progress.items():
                        latest = update_data["progress"]
                        total = update_data["total"]
                        # update the progress bar for this task:
                        progress.update(
                            task_id,
                            completed=latest,
                            total=total,
                            visible=latest < total,
                        )

                # raise any errors:
                for future in futures:
                    print(future.result())
                    
    return pd.concat([future.result() for future in futures])