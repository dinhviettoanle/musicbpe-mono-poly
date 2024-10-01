"""
Intervalize an existing MidiTok tokenization strategy
"""
from typing import Type, List, Tuple, Dict, Optional, Union, Callable
from miditoolkit import MidiFile, Instrument, Note, TempoChange
from copy import deepcopy
from operator import itemgetter

from .miditok import REMI
from .miditok.vocabulary import Vocabulary, Event
from .miditok.constants import (
    PITCH_RANGE,
    NB_VELOCITIES,
    BEAT_RES,
    ADDITIONAL_TOKENS,
    TIME_DIVISION,
    TEMPO,
    MIDI_INSTRUMENTS,
)
from .miditok.midi_tokenizer_base import MIDITokenizer, Event, Vocabulary

from .utils import token_to_midi
from .utils import *

from pprint import pprint

# Token types just before a Pitch token
SEPARATION_TOKEN_DICT = {
    'REMI': ['Position'],
    'REMIVelocityMute': ['Position'],
    'Structured': ['Time-Shift'],
    'StructuredVelocityMute': ['Time-Shift'],
    'TSD': ['Time-Shift', 'PAD'],
    'TSDVelocityMute': ['Time-Shift', 'PAD'],
}

REF_MIDI_NUMBER = 60

            
def interval_tokenizer(
        tokenizer: Type[MIDITokenizer], 
        break_simultaneous_tokens: Callable[[int, Vocabulary], bool],
        include_first_note: bool = False, 
        *args, **kwargs
    ):
    """Intervalize a monodimensional tokenization strategy. 
    Must be called as :
    ```
    interval_tokenizer(TokenizerClass, break_simultaneous_tokens_fn, include_first_note, *args, **kwargs)
    ```
    where *args and **kwargs are the arguments used to define the absolute tokenization strategy `TokenizerClass(*args, **kwargs)`

    Parameters
    ----------
    tokenizer : Type[MIDITokenizer]
        Basic Tokenizer class
    break_simultaneous_tokens : Callable[[int, Vocabulary], bool]
        Function to be called on a token id, and tell if it is the end of a chord (i.e. simulataneous notes)
    include_first_note : bool, optional
        Include the first note as absolute pitch token, by default False

    Returns
    -------
    MIDITokenizer
        Intervalized Tokenizer (of same type as tokenizer)
    
    Raises
    ------
    NotImplementedError
        Not implemented for a tokenization strategy
    """
    class IntervalTokenizer(tokenizer):
        def __init__(self):
            self.include_first_note = include_first_note
            
            # Remove useless kwargs for the base tokenizer init
            kwargs_base = kwargs.copy()
            [kwargs_base.pop(key, None) for key in ['preprocess_track_tokens', 'postprocess_track_tokens', 'skip_appending_next_token', 'postprocess_simultaneous_tokens']]
            
            super().__init__(*args, **kwargs_base)
            self.base_tokenizer = tokenizer(*args, **kwargs_base)
            #print("Interval tokenizer on:", self.base_tokenizer.__class__.__name__)
        
        
        # =======================================================================================================
        # ============================= INTERVALIZATION (Absolute -> Interval) ==================================
        # =======================================================================================================
        
        def midi_to_tokens(self, midi: MidiFile, *args, **kwargs) -> List[List[Union[int, List[int]]]]:
            """Converts a MidiFile to a list of tokens

            Parameters
            ----------
            midi : MidiFile
                Input Midi file

            Returns
            -------
            List[List[Union[int, List[int]]]]
                List of tokens
            """
            base_tokens = self.base_tokenizer.midi_to_tokens(midi, *args, **kwargs)
            return [self.intervalize_track(track_tokens) for track_tokens in base_tokens]
        
        
        def intervalize_track(self, track_tokens: List[int]) -> List[int]:
            """Intervalize a list of tokens

            Parameters
            ----------
            track_tokens : List[int]
                Input list of basic tokens (with absolute pitches)

            Returns
            -------
            List[int]
                List of tokens with converted absolute pitches into intervals

            Raises
            ------
            NotImplementedError
                Not implemented for a tokenizer class
            """
            # print_tokens(self.base_tokenizer, track_tokens[:50])
            # print("================")
            
            track_tokens = self.preprocess_track_tokens(track_tokens, self.base_tokenizer.vocab)
            
            separation_token = SEPARATION_TOKEN_DICT.get(self.base_tokenizer.__class__.__name__)
            
            if not separation_token:
                raise NotImplementedError(f"Not implemented for tokenizer of type {type(self.base_tokenizer)}")

            new_tokens = []
            highest_previous_pitch = -1
            highest_next_pitch = -1
            ii = 0
            while ii < len(track_tokens):
                token = track_tokens[ii]
                
                frmt_token = self.base_tokenizer.vocab._token_to_event[token]
                token_type = self.base_tokenizer.vocab.token_type(token)
                # print(f"{frmt_token:<30} {token_type}")
                
                if token_type in separation_token:                    
                    # Stop here and continue reading to get all the simultaneous tokens
                    simultaneous_tokens = []
                    for next_token in track_tokens[ii+1:]:
                        if break_simultaneous_tokens(next_token, self.base_tokenizer.vocab): # For example, break on a new "Position" token
                            break
                        
                        if not(self.skip_appending_next_token(next_token)):
                            simultaneous_tokens.append(next_token)
                    
                    
                    # print_tokens(self.base_tokenizer, simultaneous_tokens, RETURN_LINE=False)
                    
                    # Create verticalPitchShifts
                    ## Group pitch with their duration, velocity, etc... (i.e. break the list at each Pitch)
                    note_data_list = []
                    current_note_data = []
                    for ii_tok, sim_token in enumerate(simultaneous_tokens):
                        token_type = self.base_tokenizer.vocab.token_type(sim_token)
                        if token_type == 'Pitch' and len(current_note_data) > 0:
                            note_data_list.append(current_note_data)
                            current_note_data = []

                        if token_type == 'Pitch':
                            current_note_data.append(sim_token)
                        else:
                            current_note_data.append(self.transposed_token_base_to_interval(sim_token)) # Due to different indexing of vocab (removed tokens)
                    
                    ## Order simultaneous_tokens by pitch
                    note_data_list.append(current_note_data)
                    note_data_list_sorted = sorted(note_data_list, key=itemgetter(0), reverse=True) # False => HPitch on the bassline ; True => HPitch on the melody
                    
                    highest_previous_pitch = highest_next_pitch # The previous highest pitch was the last highest pitch
                    highest_next_pitch = note_data_list_sorted[0][0] # The highest note is necessarily the first token of the sorted simultaneous_tokens
                    
                    interval_data_list_sorted = deepcopy(note_data_list_sorted) # Same list, but with intervals

                    ## Replace simultaneous pitches with vertical pitch shifts
                    for kk, (note, next_note) in enumerate(zip(note_data_list_sorted[0:], note_data_list_sorted[1:])):
                        midi_higher_pitch = token_to_midi(self.base_tokenizer, note[0])
                        midi_lower_pitch = token_to_midi(self.base_tokenizer, next_note[0])
                        midi_interval = midi_lower_pitch - midi_higher_pitch 
                        interval_data_list_sorted[kk + 1][0] = self.get_token_from_vertical_interval(midi_interval)
                    
                    simultaneous_tokens = [item for sublist in interval_data_list_sorted for item in sublist] # Flatten the list of tuples
                                        
                    # Create horizontalPitchShifts
                    if highest_previous_pitch != -1:
                        midi_highest_previous_pitch = token_to_midi(self.base_tokenizer, highest_previous_pitch)
                        midi_highest_next_pitch = token_to_midi(self.base_tokenizer, highest_next_pitch)
                        midi_interval = midi_highest_next_pitch - midi_highest_previous_pitch
                        
                        ## First pitch (which is the highest) of the block becomes an horizontal pitch shift
                        simultaneous_tokens[0] = self.get_token_from_horizontal_interval(midi_interval)
                                        
                    
                    # Special side effect for the first pitch token
                    if highest_previous_pitch == -1 and not(self.include_first_note):
                        simultaneous_tokens.pop(0)
                        ii += 1 # bypass the pitch token in the pointer position

                    # If more than 2 notes, maybe some preprocessing to do (add Time-Shift(0) for Structured)
                    if len(note_data_list) > 1:
                        simultaneous_tokens = self.postprocess_simultaneous_tokens(simultaneous_tokens)
                    
                    # Add the position token at the beginning
                    simultaneous_tokens.insert(0, self.transposed_token_base_to_interval(token))
    
                    # Extend the list of new tokens with the intervalized block
                    new_tokens.extend(simultaneous_tokens)
                    ii += len(simultaneous_tokens)
                    
                    # print_tokens(self, simultaneous_tokens, RETURN_LINE=False)
                    # print()
                
                
                else:
                    new_tokens.append(self.transposed_token_base_to_interval(token))
                    ii += 1
            
            new_tokens = self.postprocess_track_tokens(new_tokens, self.vocab)
            return new_tokens
        
        
        def transposed_token_base_to_interval(self, token: int) -> int:
            """Convert a token ID from the base vocabulary into the token ID from the new intervalized vocabulary

            Parameters
            ----------
            token : int
                Token ID in the base vocabulary

            Returns
            -------
            int
                Token ID in the new vocabulary
            """
            event = self.base_tokenizer.vocab.token_to_event[token]
            return self.vocab.event_to_token[event]
        
        
        def _create_vocabulary(self, sos_eos_tokens: bool = None) -> Vocabulary:
            """Creates the vocabulary with interval tokens, 
            and potentially exclude the absolute pitch tokens (if not include_first_note)

            Parameters
            ----------
            sos_eos_tokens : bool, optional
                Include SOS and EOS tokens, by default None

            Returns
            -------
            Vocabulary
                Vocabulary of the intervalized tokenizer
            """
            vocab = super()._create_vocabulary(sos_eos_tokens=sos_eos_tokens)
            
            # VerticalPitchShift
            ps_index = 0
            vocab.add_event(f'VPitchShift_{ps_index}')
            for i in range(len(self.pitch_range)):
                ps_index += 1
                vocab.add_event(f'VPitchShift_{ps_index}')
                vocab.add_event(f'VPitchShift_{-ps_index}')


            # HorizontalPitchShift
            ps_index = 0
            vocab.add_event(f'HPitchShift_{ps_index}')
            for i in range(len(self.pitch_range)):
                ps_index += 1
                vocab.add_event(f'HPitchShift_{ps_index}')
                vocab.add_event(f'HPitchShift_{-ps_index}')
                        
            # Remove Pitches
            if not(self.include_first_note):
                vocab.remove_token_type('Pitch')
                
            # Reindex events
            vocab.reindex_vocab()
            
            vocab.update_token_types_indexes()
            return vocab


        def _create_token_types_graph(self) -> Dict[str, List[str]]:
            return super()._create_token_types_graph()
        
        
        def get_token_from_horizontal_interval(self, interval: int) -> int:
            """Converts an horizontal interval to its HPitchShift token

            Parameters
            ----------
            interval : int
                Horizontal interval in semitones

            Returns
            -------
            int
                Corresponding token
            """
            return self.vocab.event_to_token[f"HPitchShift_{interval}"] 
        
        
        def get_token_from_vertical_interval(self, interval: int) -> int:
            """Converts a vertical interval to its VPitchShift token

            Parameters
            ----------
            interval : int
                Vertical interval in semitones

            Returns
            -------
            int
                Corresponding token
            """
            return self.vocab.event_to_token[f"VPitchShift_{interval}"] 
        
        
        # ==== PATCHES =====
        def preprocess_track_tokens(self, track_tokens: List[int], vocab):
            """
            Preprocess the basic track before doing anything
            """
            if kwargs.get('preprocess_track_tokens'):
                return kwargs.get('preprocess_track_tokens')(track_tokens, vocab)
            return track_tokens
            
        def postprocess_track_tokens(self, track_tokens: List[int], vocab):
            """
            Postprocess the intervalized track after having done all the needed stuff 
            """
            if kwargs.get('postprocess_track_tokens'):
                return kwargs.get('postprocess_track_tokens')(track_tokens, vocab)
            return track_tokens

        def skip_appending_next_token(self, next_token: int):
            """
            Skip a token to be appended to the token list
            """
            if kwargs.get('skip_appending_next_token'):
                return kwargs.get('skip_appending_next_token')(next_token, self.base_tokenizer.vocab)
            return False
        
        def postprocess_simultaneous_tokens(self, simultaneous_tokens: List[int]):
            """
            Post process a chunk of simulatanoeus tokens (i.e. a chord)
            For example, adding Time-Shift(0) between each pitches for Structured tokenization
            """
            if kwargs.get('postprocess_simultaneous_tokens'):
                return kwargs.get('postprocess_simultaneous_tokens')(simultaneous_tokens, self.vocab)
            return simultaneous_tokens
        
        
        
        # =======================================================================================================
        # =========================== UNINTERVALIZATION (Interval -> Absolute) ==================================
        # =======================================================================================================
        
        def tokens_to_track(self, tokens: List[Union[int, List[int]]], time_division: Optional[int] = TIME_DIVISION,
                program: Optional[Tuple[int, bool]] = (0, False)) -> Tuple[Instrument, List[TempoChange]]:
            
            return self.base_tokenizer.tokens_to_track(
                tokens=self.unintervalize_track(tokens),
                time_division=time_division,
                program=program
            )
            
            
        def unintervalize_track(self, track_tokens, first_note_token=None):
            
            separation_token = SEPARATION_TOKEN_DICT.get(self.base_tokenizer.__class__.__name__)
            
            if first_note_token is None: 
                first_note_token = self.get_first_note_token(track_tokens)
            
            track_tokens = self.preprocess_track_tokens(track_tokens, self.vocab)
            
            new_tokens = []
            ii = 0
            
            ## Put first note token
            while True:
                token = track_tokens[ii]
                token_type = self.vocab.token_type(token)
                if token_type in separation_token:
                    new_tokens += [self.transposed_token_interval_to_base(token), first_note_token]
                    ii += 1
                    break
                else:
                    new_tokens.append(self.transposed_token_interval_to_base(token))
                ii += 1
            
            # print("Preprocessing")
            # print_tokens(self.base_tokenizer, new_tokens)
            # print()
            
            # Loop on the whole sequence
            current_midi_pitch = token_to_midi(self.base_tokenizer, first_note_token)
            current_melodic_pitch = token_to_midi(self.base_tokenizer, first_note_token)
            while ii < len(track_tokens):
                token = track_tokens[ii]
                
                frmt_token = self.vocab._token_to_event[token]
                token_type = self.vocab.token_type(token)
                
                if "PitchShift" in token_type:
                    # print(current_midi_pitch, self.get_interval_from_pitchshift_token(token), frmt_token)
                    if token_type == "VPitchShift":
                        current_midi_pitch = current_midi_pitch + self.get_interval_from_pitchshift_token(token)
                    else:
                        current_midi_pitch = current_melodic_pitch + self.get_interval_from_pitchshift_token(token)
                        current_melodic_pitch = current_melodic_pitch + self.get_interval_from_pitchshift_token(token)
                    
                    try:
                        next_token = midi_to_token(self.base_tokenizer, current_midi_pitch)
                    except IndexError:
                        print(f"WARNING: Pitch token not found for MIDI number = {current_midi_pitch}")
                        next_token = 0
                        
                    new_tokens.append(next_token)
                
                elif token_type != 'Pitch': # Not add a duplicate of first_pitch if it is available
                    new_tokens.append(self.transposed_token_interval_to_base(token))

                if new_tokens[-1] is None:
                    print(frmt_token, current_midi_pitch)
                    exit()
                ii += 1
                
            new_tokens = self.postprocess_track_tokens(new_tokens, self.base_tokenizer.vocab)
            new_tokens = self.add_velocities(new_tokens)
            
            print("============= FINAL =========")
            print_tokens(self.base_tokenizer, new_tokens)
            print()
            
            return new_tokens
        
        
        def transposed_token_interval_to_base(self, token: int) -> int:
            """Convert a token ID from the base vocabulary into the token ID from the new intervalized vocabulary

            Parameters
            ----------
            token : int
                Token ID in the base vocabulary

            Returns
            -------
            int
                Token ID in the new vocabulary
            """
            event = self.vocab.token_to_event[token]
            return self.base_tokenizer.vocab.event_to_token[event]


        def get_first_note_token(self, track_tokens):
            """Set the first note of the converted intervalized sequence
            If the first note is already included in the tokenization strategy, use this note
            Else, take `REF_MIDI_NUMBER` as an arbitrary beginning note
            
            Warning: the resulting sequence can be too low or too high in the MIDI scale: 
            these overflows are replaced by PAD tokens

            Parameters
            ----------
            track_tokens : List[int]
                List of tokens

            Returns
            -------
            int
                Token ID of the first note
            """
            if not(include_first_note):
                return midi_to_token(self.base_tokenizer, REF_MIDI_NUMBER)
            else:
                return next(t for t in track_tokens if self.vocab.token_type(t) == 'Pitch')
        
        
        def get_interval_from_pitchshift_token(self, token):
            """Get interval as integer from a token ID

            Parameters
            ----------
            token : int
                Token ID

            Returns
            -------
            int
                Interval in semitones
            """
            frmt_token = self.vocab._token_to_event[token]
            return int(frmt_token.split('_')[1])
        
        
        def add_velocities(self, track_tokens):
            """
            Post process a chunk of simulatanoeus tokens (i.e. a chord)
            For example, adding Time-Shift(0) between each pitches for Structured tokenization
            """
            if kwargs.get('add_velocities'):
                return kwargs.get('add_velocities')(track_tokens, self.base_tokenizer.vocab)
            return track_tokens

    
    return IntervalTokenizer()



