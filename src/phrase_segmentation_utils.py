"""
Phrase segmentation utils functions
(mainly polyphonic - for monophonic, look at notebooks/experiment_211_231.ipynb)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import time
import pprint
pp = pprint.PrettyPrinter(width=20)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, matthews_corrcoef, balanced_accuracy_score
from torch.utils.data import random_split
import sklearn.metrics as metrics


from .utils import print_tokens, tokens_to_bow
from .tokenizer import *


tqdm.pandas()



def decompose_bpe_align(original_tokens, tokenizer_bpe, element_to_insert=0):
    """Align original tokens list with its BPE-tokenized version with supertokens
    Supertokens are derived as <ID,0,0,0>, if the supertoken ID has 4 atomic elements

    Parameters
    ----------
    original_tokens : List[int]
        List of tokens as atomic elements
    tokenizer_bpe : MIDITokenizer
        BPE-trained tokenizer

    Returns
    -------
    List[int], List[int]
        Initial list of atomic elements, List of aligned supertokens
    """
    i = 0
    tokens = original_tokens.copy()
    aligned_supertokens = original_tokens.copy()
    while i < len(tokens):
        current_token = int(tokens[i])
        token_type, token_val = tokenizer_bpe.vocab[current_token].split('_')
        if token_type == 'BPE':
            del tokens[i]
            del aligned_supertokens[i]
            aligned_supertokens.insert(i, current_token)
            for j, to_insert in enumerate(map(int, token_val.split('.')[1].split('-'))):
                tokens.insert(i + j, to_insert)
                if j > 0:
                    if element_to_insert == 'length':
                        fill_in_value = len(token_val.split('.')[1].split('-'))
                    else:
                        fill_in_value = 0
                    aligned_supertokens.insert(i + j, fill_in_value) # current_token
        i += 1
    return tokens, aligned_supertokens


def decompose_sanity_align(original_tokens, regular_split_vocab):
    i = 0
    tokens = original_tokens.copy()
    aligned_supertokens = []
    while i < len(tokens):
        current_token = int(tokens[i])
        elementary_tokens = regular_split_vocab[current_token].split('-')
        aligned_supertokens.append(current_token)
        aligned_supertokens.extend([0]*(len(elementary_tokens) - 1))
        i += 1
    return tokens, aligned_supertokens


def compute_start_of_phrase_window_bpe(start_of_phrase_window, aligned_tokens_bpe, tokens_bpe):
    # WARNING: not sure it works all the time (if the start-of-phrase is in the middle of a supertoken)
    # Not working with one-hot column !
    start_of_phrase_window_bpe = np.zeros(len(tokens_bpe), dtype=float)
    pointer_bpe = 0
    pointer_no_bpe = 0
    
    while pointer_no_bpe < len(aligned_tokens_bpe):
        # print(pointer_bpe, len(tokens_bpe), tokens_bpe[pointer_bpe], aligned_tokens_bpe[pointer_no_bpe])
        if tokens_bpe[pointer_bpe] == aligned_tokens_bpe[pointer_no_bpe]:
            start_of_phrase_window_bpe[pointer_bpe] = start_of_phrase_window[pointer_no_bpe]
            pointer_bpe += 1 if pointer_bpe < len(tokens_bpe)-1 else 0
        pointer_no_bpe += 1

    return start_of_phrase_window_bpe



def compute_start_of_phrase_from_one_hot(start_of_phrase_onehot, aligned_tokens_bpe, tokens_bpe, supertokens_set):
    # Works both with window and one-hot
    start_of_phrase_nonbpe = []
    start_of_phrase_bpe = []
    pointer_non_bpe = 0
    pointer_bpe = 0
    
    while pointer_non_bpe < len(aligned_tokens_bpe):
        if aligned_tokens_bpe[pointer_non_bpe] in supertokens_set:
            length_supertoken = aligned_tokens_bpe[pointer_non_bpe + 1]
            assert length_supertoken > 0
            
            block_start_of_phrase = start_of_phrase_onehot[pointer_non_bpe:pointer_non_bpe+length_supertoken]
            start_of_phrase_value = int(1 in block_start_of_phrase)
            
            start_of_phrase_nonbpe.extend([start_of_phrase_value] * length_supertoken)
            start_of_phrase_bpe.append(start_of_phrase_value)
            
            pointer_non_bpe += length_supertoken
            pointer_bpe += 1
        else:
            start_of_phrase_nonbpe.append(start_of_phrase_onehot[pointer_non_bpe])
            start_of_phrase_bpe.append(start_of_phrase_onehot[pointer_non_bpe])
            pointer_non_bpe += 1
            pointer_bpe += 1
            

    assert len(start_of_phrase_nonbpe) == len(aligned_tokens_bpe)
    assert len(start_of_phrase_bpe) == len(tokens_bpe)
    
    return start_of_phrase_nonbpe, start_of_phrase_bpe



def make_start_of_phrase_onehot(lst):
    arr = np.array(lst)
    start_of_phrase_arr = np.zeros_like(arr)
    change_indices = np.where(arr[:-1] != arr[1:])[0] + 1
    start_of_phrase_arr[change_indices] = 1
    start_of_phrase_arr[0] = 1
    return start_of_phrase_arr


def make_start_of_phrase_window(lst, window_size, with_zero=True):
    arr = np.array(lst)
    len_arr = len(arr)
    start_of_phrase_arr = np.zeros_like(arr)
    change_indices = np.where(arr[:-1] != arr[1:])[0] + 1
    change_indices = [0] + list(change_indices) if with_zero else list(change_indices)
    for idx in change_indices:
        beg_window = max(0, idx - window_size // 2)
        end_window = min(len_arr, idx + window_size // 2)
        start_of_phrase_arr[beg_window:end_window] = 1
    return start_of_phrase_arr

def gaussian(x, mu, sig):
    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )


def make_start_of_phrase_gaussian(lst, gaussian_std, with_zero=True):
    arr = np.array(lst)
    len_arr = len(arr)
    start_of_phrase_arr = np.zeros_like(arr, dtype=float)
    change_indices = np.where(arr[:-1] != arr[1:])[0] + 1
    change_indices = [0] + list(change_indices) if with_zero else list(change_indices)
    for idx in change_indices:
        gaussian_reg = gaussian(np.arange(len(arr)), idx, gaussian_std)
        start_of_phrase_arr += gaussian_reg / gaussian_reg.max()
    return start_of_phrase_arr


def check_balanced_accuracy(df_token_phrase, column_start_of_phrase):
        number_no_end = sum(df_token_phrase[column_start_of_phrase].apply(lambda x: np.count_nonzero(np.array(x) == 0)))
        number_with_end = sum(df_token_phrase[column_start_of_phrase].apply(lambda x: np.count_nonzero(np.array(x) == 1)))
        total_tags = number_no_end + number_with_end
        print("Balanced dataset: [0] {:.2f} ({:,}) / [1] {:.2} ({:,}) ; Total: {:,}".format(
            number_no_end / total_tags, number_no_end, number_with_end / total_tags, number_with_end, total_tags
        ))

    


class PhraseDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, label, 
                 max_len=512, token_pad=0, tag_pad=2
                ):
        self.tokens = tokens
        self.label = label
        self.max_length = max_len
        self.token_pad = token_pad
        self.tag_pad = tag_pad
    
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, item):
        toks = self.tokens[item]
        label = self.label[item]

        if len(toks) > self.max_length:
            toks = toks[:self.max_length]
            label = label[:self.max_length]
        
        
        ########################################
        # Forming the inputs
        ids = toks
        att_mask = [1] * len(ids)
        
        # Padding
        pad_len = self.max_length - len(ids)        
        ids = list(ids) + [self.token_pad] * pad_len
        label = list(label) + [self.tag_pad] * pad_len
        att_mask = att_mask + [0] * pad_len
                
        return {
            'ids': torch.tensor(ids, dtype = torch.long),
            'att_mask': torch.tensor(att_mask, dtype = torch.long),
            'target': torch.tensor(label)
        }
        

def create_subsets(dataset, split_ratio):
    """Create subsets of a dataset following split ratios.
    if sum(split_ratio) != 1, the remaining portion will be inserted as the first subset

    Parameters
    ----------
    dataset : torch.Dataset
        Dataset object, must implement the __len__ magic method.
    split_ratio : float
        Split ratios as a list of float

    Returns
    -------
    torch.Tensor
        The list of subsets
    """
    
    assert all(0 <= ratio <= 1. for ratio in split_ratio), 'The split ratios must be comprise within [0,1]'
    assert sum(split_ratio) <= 1., 'The sum of split ratios must be inferior or equal to 1'
    len_subsets = [int(len(dataset) * ratio) for ratio in split_ratio]
    if sum(split_ratio) != 1.:
        len_subsets.insert(0, len(dataset) - sum(len_subsets))
    subsets = random_split(dataset, len_subsets)
    return subsets


# ======================================================================================================================

def compute_classif_perfo(preds, targets, thr=0.5):
    true_labels, predictions = [], []

    for ii in range(len(targets)):
        clamp_preds = preds[ii].copy()
        clamp_preds[clamp_preds > thr] = 1
        clamp_preds[clamp_preds <= thr] = 0

        clamp_targets = targets[ii].copy()
        clamp_targets[clamp_targets > thr] = 1
        clamp_targets[clamp_targets <= thr] = 0
        
        predictions.append(list(clamp_preds))
        true_labels.append(list(clamp_targets))
        
    true_labels = np.concatenate(true_labels)
    predictions = np.concatenate(predictions)

    dict_perfo = {
        'test_acc': metrics.accuracy_score(y_true=true_labels, y_pred=predictions),
        'test_recall': metrics.recall_score(y_true=true_labels, y_pred=predictions, average='macro', labels=[0, 1]),
        'test_precision': metrics.precision_score(y_true=true_labels, y_pred=predictions, average='macro', labels=[0, 1]),
        'test_f1_macro': metrics.f1_score(y_true=true_labels, y_pred=predictions, average='macro', labels=[0, 1]),
        'test_f1weighted': metrics.f1_score(y_true=true_labels, y_pred=predictions, average='weighted', labels=[0, 1]),
    }
    
    print(metrics.classification_report(y_true=true_labels, y_pred=predictions, labels=[0, 1]))
    
    return dict_perfo



# =============================================================================
# ==================== OVERLAPPING SUPERTOKENS ================================
# =============================================================================


def count_overlapping_supertokens(indexes_phrases, aligned_supertokens, full_piece_tokens, window=5, VERBOSE=True, tokenizer_bpe=None):
    """Count number of overlapping supertokens within a piece

    Parameters
    ----------
    indexes_phrases : List[int]
        List of phrase indexes of the piece (e.g. [0,0,0,0,1,1,1,1,2,2,2])
    aligned_supertokens : List[int]
        List of tokens of the piece as aligned supertokens
    full_piece_tokens : List[int]
        List of tokens of the full piece as atomic elements
    window : int, optional
        Window to show the phrase breaks (only useful when VERBOSE=True), by default 5
    VERBOSE : bool, optional
        Show the phrase breaks and the neighbor tokens, by default True
    tokenizer_bpe : _type_, optional
        BPE-trained tokenizer (only useful when VERBOSE=True), by default None

    Returns
    -------
    int, int
        Number of overlapping supertokens, Number of phrases
    """
    overlapping_supertokens = 0
    n_phrases = 0
    for ii in range(len(indexes_phrases)-1):
        if indexes_phrases[ii] != indexes_phrases[ii + 1]:
            chunk_phrases = indexes_phrases[ii-window+1:ii+window]
            chunk_supertokens = aligned_supertokens[ii-window+1:ii+window]
            chunk_original = full_piece_tokens[ii-window+1:ii+window]

            if VERBOSE:
                pp.pprint(list(zip(chunk_phrases, chunk_supertokens)))
                print(aligned_supertokens[ii] == aligned_supertokens[ii + 1])
                print("***")
                print_tokens(tokenizer_bpe, chunk_supertokens)
                print("***")
                print_tokens(tokenizer_bpe, chunk_original)
                print("=====")


            if aligned_supertokens[ii] == aligned_supertokens[ii + 1]:
                overlapping_supertokens += 1
            n_phrases += 1
    
    if VERBOSE:
        print("Overlapping supertokens:", overlapping_supertokens, "on", n_phrases, "phrases") 
    return overlapping_supertokens, n_phrases



def random_split_test(df_phrase_data, full_piece_tokens, aligned_supertokens):
    """Split randomly a piece and count the number of overlapping supertokens

    Parameters
    ----------
    df_phrase_data : pandas.DataFrame
        DataFrame containing phrase data (computed by make_df_phrase_data)
    full_piece_tokens : List[int]
        List of tokens of the piece as atomic elements
    aligned_supertokens : List[int]
        List of tokens of the piece as supertokens

    Returns
    -------
    int
        Ratio of overlapping supertokens with a random split
    """
    list_random_overlapping = []

    for _ in range(100):
        random_splits_idxs = sorted(np.random.randint(0, len(full_piece_tokens), len(df_phrase_data)-1))
        current_idx_phrase = 0
        random_indexes_phrases = []
        for ii in range(len(full_piece_tokens)):
            if ii in random_splits_idxs:
                current_idx_phrase += 1
            random_indexes_phrases.append(current_idx_phrase)

        assert len(aligned_supertokens) == len(random_indexes_phrases)

        overlap_count = count_overlapping_supertokens(random_indexes_phrases, aligned_supertokens, full_piece_tokens, VERBOSE=False)
        list_random_overlapping.append(overlap_count)

    print(f"On average with random phrase splits : {np.mean(list_random_overlapping):.3f} overlapping supertokens on {len(df_phrase_data)-1} phrases")
    return np.mean(list_random_overlapping)


# =============================================================================
# ==================== CLASSIFICATION =========================================
# =============================================================================


def make_df_window_data_from_piece(tokenizer, full_piece_tokens, indexes_phrases, piece_id, window_size=75, has_end_of_phrase_fn=None):
    """Split a piece into windows to be classified and store them in a pandas.DataFrame

    Parameters
    ----------
    tokenizer : MIDITokenizer
        Tokenizer
    full_piece_tokens : List[int]
        List of tokens of a piece as atomic elements
    indexes_phrases : List[int]
        List of phrase indexes of the piece (e.g. [0,0,0,0,1,1,1,1,2,2,2])
    piece_id : str
        ID of the piece
    window_size : int, optional
        Size of the window, by default 75
    has_end_of_phrase_fn : _type_, optional
        Function to check if here is an end of phrase, by default None

    Returns
    -------
    pandas.DataFrame
        DataFrame containing data regarding the windows to be classified
    """
    list_window_data = []

    step = window_size
    for ii_beg in range(0, len(full_piece_tokens) - window_size, step):
        chunk_phrases = indexes_phrases[ii_beg:ii_beg+window_size]
        chunk_phrases_norm = chunk_phrases - np.min(chunk_phrases)
        chunk_original = full_piece_tokens[ii_beg:ii_beg+window_size]
        if has_end_of_phrase_fn is None:
            has_end_of_phrase = not(all([elt == chunk_phrases[0] for elt in chunk_phrases]))
        else:
            has_end_of_phrase = has_end_of_phrase(chunk_phrases)
        
        beging_phrase_token = np.zeros(len(chunk_phrases_norm), dtype=int)

        if has_end_of_phrase:
            beging_phrase_token[np.where(chunk_phrases_norm == 1)[0][0]] = 1

        list_window_data.append({
            'piece_id' : piece_id,
            'chunk_phrase': beging_phrase_token,
            'chunk_tokens': chunk_original,
            'has_end_of_phrase': has_end_of_phrase
        })

    df_window_data = pd.DataFrame(list_window_data)
    df_window_data['bow'] = df_window_data.chunk_tokens.apply(lambda tokens: tokens_to_bow(tokenizer, tokens))
    
    return df_window_data



def train_test_k_fold(df_train, df_test, test_idx, in_var='bow', X_train=None, X_test=None):
    """Train and evaluate a fold in a cross-validation

    Parameters
    ----------
    df_train : pandas.DataFrame
        DataFrame containing windows data for training
    df_test : pandas.DataFrame
        DataFrame containing windows data for testing
    test_idx : str
        Piece IDs used in the test set
    in_var : str, optional
        Variable to be used as input for the classifier, by default 'bow' (or can be 'bow_bpe')
    X_train : np.array, optional
        Already processed training data, by default None
    X_test : np.array, optional
        Already processed test data, by default None

    Returns
    -------
    dict
        Evaluation metrics (train accuracy, test accuracy, f1 macro, f1 weighted, mcc, balanced accuracy)
    """
    print("Splitting...")
    if X_train is None:
        X_train = np.array(df_train[in_var].tolist())
    if X_test is None:
        X_test = np.array(df_test[in_var].tolist())
    
    y_train = np.array(df_train.has_end_of_phrase.tolist())
    y_test = np.array(df_test.has_end_of_phrase.tolist())
    
    
    print("Fitting...")
    tic = time.time()
    # clf = LogisticRegression(max_iter=5000, random_state=0)
    clf = MultinomialNB()
    # clf = DummyClassifier()
    
    clf.fit(X_train, y_train)
    print("Elapsed time:", time.time() - tic)
    y_pred = clf.predict(X_test)
    
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    
    print("Train acc :", train_acc)
    print("Test acc :", test_acc)
    print("Test macro f1-score :", f1_macro)
    print("Test weighted f1-score :", f1_weighted)
    print("MCC :", mcc)
    print("Balanced accuracy :", balanced_accuracy)
    print(classification_report(y_test, y_pred))
    
    return {
        'test_idx': test_idx,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'mcc': mcc,
        'balanced_accuracy': balanced_accuracy,
    }