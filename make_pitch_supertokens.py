"""
Output:
- Data for figure 4 (supertokens with pitch)

Required:
- a pretrained BPE tokenizer
"""
import argparse
import pickle
import numpy as np
from src.tokenizer import REMIVelocityMute
from src.miditok import bpe
from src.utils import convert_bpe_to_sequence_frmt, print_vocab
from tqdm.auto import tqdm

TokenizerClass = REMIVelocityMute
nb_velocities = 1
max_pitches = 120
max_steps = 32000

def count_pitch_tokens(root_events):
    types = {
        'pitch': 0,
        'duration': 0,
        'position': 0
    }
    for tok in root_events:
        if 'Pitch' in tok: 
            types['pitch'] += 1
        if 'Duration' in tok:
            types['duration'] += 1
        if 'Position' in tok:
            types['position'] += 1
    return types

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bpe_tokenizer', default='bpe_tokenizers/REMIVelocityMute_128000_mtcpiano.bpe')
    parser.add_argument('--outfile', required=False, default='mtcmono_by_types.pickle')
    args = parser.parse_args()
    
    print("BPE tokenizer path:", args.bpe_tokenizer)
    print("Out pickle file:", args.outfile)
  
    tokenizer_bpe = bpe(TokenizerClass, nb_velocities=nb_velocities)
    tokenizer_bpe.load_params(args.bpe_tokenizer)

    # print_vocab(tokenizer_bpe)
    print("Vocab: {}".format(len(tokenizer_bpe)))
    
    by_step_type_tokens = []

    for token in tokenizer_bpe.vocab._token_to_event:
        frmt_token = tokenizer_bpe.vocab._token_to_event[token]
        if tokenizer_bpe.vocab.token_type(token) == 'BPE':
            frmt_root_events = convert_bpe_to_sequence_frmt(tokenizer_bpe, frmt_token, AS_MUSIC=False)
            by_step_type_tokens.append(count_pitch_tokens(frmt_root_events))
            # print(token, ":", f"<BPE {frmt_root_events}>")
        # else:
        #     print(token, ":", frmt_token)
        
        
    by_step_cumulative_type_tokens = []
    all_count_pitches = np.zeros((max_steps, max_pitches))
    all_count_duration = np.zeros((max_steps, max_pitches))
    all_count_position = np.zeros((max_steps, max_pitches))

    for step, types_tokens in tqdm(enumerate(by_step_type_tokens[:max_steps]), total=max_steps):
        count_pitches = np.zeros(max_pitches)
        count_duration = np.zeros(max_pitches)
        count_position = np.zeros(max_pitches)
        for ii in range(0, step):
            n_pitches = by_step_type_tokens[ii]['pitch']
            n_durations = by_step_type_tokens[ii]['duration']
            n_positions = by_step_type_tokens[ii]['position']
            count_pitches[n_pitches] += 1
            count_duration[n_durations] += 1
            count_position[n_positions] += 1
        
        all_count_pitches[step] = count_pitches
        all_count_duration[step] = count_duration
        all_count_position[step] = count_position
        
    results = {
        'pitches': all_count_pitches,
        'durations': all_count_duration,
        'positions': all_count_position,
        'norm_matrix': np.tile(np.arange(max_steps), (max_pitches, 1)).T,
        'max_steps': max_steps,
    }
    with open(args.outfile, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
