"""
Custom tokenization strategies:
- REMIVelocityMute: REMI with Velocity tokens being removed
- REMIInterval: REMI Interval, with Velocity tokens being kept
- REMIIntervalVelocityMute: REMI Intervalized, with Velocity tokens being removed
- StructuredVelocityMute: Structured with Velocity tokens being removed
- StructuredInterval: Structured Intervalized, with Velocity tokens being removed
- StructuredIntervalVelocityMute: Structured Intervalized, with Velocity tokens being removed
- TSDVelocityMute: TSD with Velocity tokens being removed
- TSDIntervalVelocityMute: TSD Intervalized, with Velocity tokens being removed

BUG: Why can't parallel_apply when StructuredInterval ?
"""
from typing import List, Tuple, Dict, Optional, Union
from miditoolkit import MidiFile, Instrument, Note, TempoChange

from .miditok import REMI, Structured, TSD
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

from .interval_tokenizer import interval_tokenizer


# ======================================================================================================

class REMIVelocityMute(REMI):
    def __init__(
        self,
        pitch_range: range = PITCH_RANGE,
        beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
        nb_velocities: int = NB_VELOCITIES,
        additional_tokens: Dict[str, Union[bool, int]] = ADDITIONAL_TOKENS,
        pad: bool = True,
        sos_eos: bool = False,
        mask: bool = False,
        params=None,
        *args, **kwargs,
    ):
        # No additional tokens
        additional_tokens["Chord"] = False  # Incompatible additional token
        additional_tokens["Rest"] = False
        additional_tokens["Tempo"] = False
        additional_tokens["TimeSignature"] = False
        super().__init__(
            pitch_range, beat_res,
            1,  # nb_velocities
            additional_tokens, pad, sos_eos, mask, params=params,
        )

    def track_to_tokens(self, track: Instrument) -> List[int]:
        remi_tokens = super().track_to_tokens(track)
        new_tokens = []
        for token in remi_tokens:
            if self.vocab.token_type(token) != "Velocity":
                new_tokens.append(token)
        return new_tokens

    def _create_vocabulary(self, sos_eos_tokens: bool = None) -> Vocabulary:
        vocab = super()._create_vocabulary(sos_eos_tokens=sos_eos_tokens)
        return vocab
        
# ======================================================================================================

class StructuredVelocityMute(Structured):
    def __init__(
        self,
        pitch_range: range = PITCH_RANGE,
        beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
        nb_velocities: int = NB_VELOCITIES,
        additional_tokens: Dict[str, Union[bool, int]] = ADDITIONAL_TOKENS,
        pad: bool = True,
        sos_eos: bool = False,
        mask: bool = False,
        params=None,
        *args, **kwargs,
    ):
        # No additional tokens
        additional_tokens["Chord"] = False  # Incompatible additional token
        additional_tokens["Rest"] = False
        additional_tokens["Tempo"] = False
        additional_tokens["TimeSignature"] = False
        super().__init__(
            pitch_range, beat_res,
            1, # nb_velocities
            additional_tokens, pad, sos_eos, mask, params=params,
        )

    def track_to_tokens(self, track: Instrument) -> List[int]:
        structured_tokens = super().track_to_tokens(track)
        new_tokens = []
        for token in structured_tokens:
            if self.vocab.token_type(token) != "Velocity":
                new_tokens.append(token)
        return new_tokens

    def _create_vocabulary(self, sos_eos_tokens: bool = None) -> Vocabulary:
        vocab = super()._create_vocabulary(sos_eos_tokens=sos_eos_tokens)
        return vocab


# ======================================================================================================
# ======================================================================================================


class REMIIntervalVelocityMute(REMI):
    def __init__(
        self,
        include_first_note: bool = False,
        pitch_range: range = PITCH_RANGE,
        beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
        nb_velocities: int = NB_VELOCITIES,
        additional_tokens: Dict[str, Union[bool, int]] = ADDITIONAL_TOKENS,
        pad: bool = True,
        sos_eos: bool = False,
        mask: bool = False,
        params=None,
    ):
        # No additional tokens
        additional_tokens["Chord"] = False  # Incompatible additional token
        additional_tokens["Rest"] = False
        additional_tokens["Tempo"] = False
        additional_tokens["TimeSignature"] = False
        self.remi_interval_tokenizer = interval_tokenizer(
            REMIVelocityMute,
            break_simultaneous_tokens=self.break_simultaneous_tokens,
            include_first_note=include_first_note,
            add_velocities=self.add_velocities,
            pitch_range=pitch_range,
            beat_res=beat_res,
            nb_velocities=1,
            additional_tokens=additional_tokens,
            pad=pad,
            sos_eos=sos_eos,
            mask=mask,
            params=params,
        )
        
        super().__init__(pitch_range, beat_res, 1, additional_tokens, pad, sos_eos, mask, params=params)
        
    def midi_to_tokens(self, midi: MidiFile, *args, **kwargs) -> List[List[Union[int, List[int]]]]:
        return self.remi_interval_tokenizer.midi_to_tokens(midi, *args, **kwargs)

    def _create_vocabulary(self, sos_eos_tokens: bool = None) -> Vocabulary:
        vocab = self.remi_interval_tokenizer.vocab
        vocab.remove_token_type('Velocity')
        return vocab
    
    @staticmethod
    def break_simultaneous_tokens(next_token, base_tokenizer_vocab):
        next_token_type = base_tokenizer_vocab.token_type(next_token)
        return next_token_type in ['Position', 'Bar']
    
    def tokens_to_midi(self, tokens: List[List[Union[int, List[int]]]],
            programs: Optional[List[Tuple[int, bool]]] = None, output_path: Optional[str] = None,
            time_division: Optional[int] = TIME_DIVISION) -> MidiFile:
        return self.remi_interval_tokenizer.tokens_to_midi(tokens, programs, output_path, time_division)
    
    def add_velocities(self, track_tokens, base_tokenizer_vocab):
        for token in base_tokenizer_vocab._token_to_event:
            frmt_token = base_tokenizer_vocab._token_to_event[token]
            if frmt_token == 'Velocity_127':
                velocity_token = token
        new_tokens = []
        for token in track_tokens:
            if base_tokenizer_vocab.token_type(token) == 'Pitch':
                new_tokens += [token, velocity_token]
            else:
                new_tokens.append(token)
        return new_tokens
        
    

# ======================================================================================================


class REMIInterval(REMI):
    def __init__(
        self,
        include_first_note: bool = False,
        pitch_range: range = PITCH_RANGE,
        beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
        nb_velocities: int = NB_VELOCITIES,
        additional_tokens: Dict[str, Union[bool, int]] = ADDITIONAL_TOKENS,
        pad: bool = True,
        sos_eos: bool = False,
        mask: bool = False,
        params=None,
    ):
        # No additional tokens
        additional_tokens["Chord"] = False  # Incompatible additional token
        additional_tokens["Rest"] = False
        additional_tokens["Tempo"] = False
        additional_tokens["TimeSignature"] = False
        self.remi_interval_tokenizer = interval_tokenizer(
            REMI,
            break_simultaneous_tokens=self.break_simultaneous_tokens,
            include_first_note=include_first_note,
            pitch_range=pitch_range,
            beat_res=beat_res,
            nb_velocities=nb_velocities,
            additional_tokens=additional_tokens,
            pad=pad,
            sos_eos=sos_eos,
            mask=mask,
            params=params,
        )
        
        super().__init__(pitch_range, beat_res, nb_velocities, additional_tokens, pad, sos_eos, mask, params=params)
        
    def midi_to_tokens(self, midi: MidiFile, *args, **kwargs) -> List[List[Union[int, List[int]]]]:
        return self.remi_interval_tokenizer.midi_to_tokens(midi, *args, **kwargs)

    def _create_vocabulary(self, sos_eos_tokens: bool = None) -> Vocabulary:
        return self.remi_interval_tokenizer.vocab
    
    @staticmethod
    def break_simultaneous_tokens(next_token, base_tokenizer_vocab):
        next_token_type = base_tokenizer_vocab.token_type(next_token)
        return next_token_type in ['Position', 'Bar']
    
    def tokens_to_midi(self, tokens: List[List[Union[int, List[int]]]],
            programs: Optional[List[Tuple[int, bool]]] = None, output_path: Optional[str] = None,
            time_division: Optional[int] = TIME_DIVISION) -> MidiFile:
        return self.remi_interval_tokenizer.tokens_to_midi(tokens, programs, output_path, time_division)
    

    

# ======================================================================================================


class StructuredInterval(Structured):
    def __init__(
        self,
        include_first_note: bool = False,
        pitch_range: range = PITCH_RANGE,
        beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
        nb_velocities: int = NB_VELOCITIES,
        additional_tokens: Dict[str, Union[bool, int]] = ADDITIONAL_TOKENS,
        pad: bool = True,
        sos_eos: bool = False,
        mask: bool = False,
        params=None,
    ):
        # No additional tokens
        additional_tokens["Chord"] = False  # Incompatible additional token
        additional_tokens["Rest"] = False
        additional_tokens["Tempo"] = False
        additional_tokens["TimeSignature"] = False
        self.structured_interval_tokenizer = interval_tokenizer(
            Structured,
            break_simultaneous_tokens=self.break_simultaneous_tokens,
            preprocess_track_tokens=self.preprocess_track_tokens,
            postprocess_track_tokens=self.postprocess_track_tokens,
            skip_appending_next_token=self.skip_appending_next_token,
            postprocess_simultaneous_tokens=self.postprocess_simultaneous_tokens,
            include_first_note=include_first_note,
            pitch_range=pitch_range,
            beat_res=beat_res,
            nb_velocities=nb_velocities,
            additional_tokens=additional_tokens,
            pad=pad,
            sos_eos=sos_eos,
            mask=mask,
            params=params,
        )
        
        super().__init__(pitch_range, beat_res, nb_velocities, additional_tokens, pad, sos_eos, mask, params=params)
    
    def midi_to_tokens(self, midi: MidiFile, *args, **kwargs) -> List[List[Union[int, List[int]]]]:
        return self.structured_interval_tokenizer.midi_to_tokens(midi, *args, **kwargs)

    def _create_vocabulary(self, sos_eos_tokens: bool = None) -> Vocabulary:
        return self.structured_interval_tokenizer.vocab
    
    @staticmethod
    def break_simultaneous_tokens(next_token, base_tokenizer_vocab):
        next_token_type = base_tokenizer_vocab.token_type(next_token)
        return next_token_type == 'Time-Shift' and \
            base_tokenizer_vocab.token_to_event[next_token] != "Time-Shift_0.0.1"
            
    @staticmethod
    def preprocess_track_tokens(track_tokens, base_tokenizer_vocab):
        # Add a dummy Time-Shift token to trigger the simulateneous tokens
        if base_tokenizer_vocab.token_type(track_tokens[0]) != 'Time-Shift': # If the first token is not already a rest
            track_tokens.insert(0, base_tokenizer_vocab.event_to_token["Time-Shift_0.0.1"])
        return track_tokens
    
    @staticmethod
    def postprocess_track_tokens(track_tokens, self_vocab):
        # Remove the first dummy Time-Shift token
        if self_vocab.token_to_event[track_tokens[0]] == "Time-Shift_0.0.1": # If the first token wasn't already a rest
            track_tokens.pop(0)
        return track_tokens
    
    @staticmethod
    def skip_appending_next_token(next_token, base_tokenizer_vocab):
        next_token_type = base_tokenizer_vocab.token_type(next_token)
        return next_token_type == 'Time-Shift'
    
    @staticmethod
    def postprocess_simultaneous_tokens(simultaneous_tokens, self_vocab):
        temp_simultaneous_tokens = []
        TS_0_token = self_vocab.event_to_token["Time-Shift_0.0.1"]
        for token in simultaneous_tokens:
            token_type = self_vocab.token_type(token)
            if token_type == 'VPitchShift':
                temp_simultaneous_tokens += [TS_0_token, token]
            else:
                temp_simultaneous_tokens.append(token)
        return temp_simultaneous_tokens
    
    def tokens_to_midi(self, tokens: List[List[Union[int, List[int]]]],
            programs: Optional[List[Tuple[int, bool]]] = None, output_path: Optional[str] = None,
            time_division: Optional[int] = TIME_DIVISION) -> MidiFile:
        return self.structured_interval_tokenizer.tokens_to_midi(tokens, programs, output_path, time_division)

    

# ======================================================================================================


class StructuredIntervalVelocityMute(Structured):
    def __init__(
        self,
        include_first_note: bool = False,
        pitch_range: range = PITCH_RANGE,
        beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
        nb_velocities: int = NB_VELOCITIES,
        additional_tokens: Dict[str, Union[bool, int]] = ADDITIONAL_TOKENS,
        pad: bool = True,
        sos_eos: bool = False,
        mask: bool = False,
        params=None,
    ):
        # No additional tokens
        additional_tokens["Chord"] = False  # Incompatible additional token
        additional_tokens["Rest"] = False
        additional_tokens["Tempo"] = False
        additional_tokens["TimeSignature"] = False
        self.structured_interval_tokenizer = interval_tokenizer(
            StructuredVelocityMute,
            break_simultaneous_tokens=self.break_simultaneous_tokens,
            preprocess_track_tokens=self.preprocess_track_tokens,
            postprocess_track_tokens=self.postprocess_track_tokens,
            skip_appending_next_token=self.skip_appending_next_token,
            postprocess_simultaneous_tokens=self.postprocess_simultaneous_tokens,
            add_velocities=self.add_velocities,
            include_first_note=include_first_note,
            pitch_range=pitch_range,
            beat_res=beat_res,
            nb_velocities=1,
            additional_tokens=additional_tokens,
            pad=pad,
            sos_eos=sos_eos,
            mask=mask,
            params=params,
        )
        
        super().__init__(pitch_range, beat_res, 1, additional_tokens, pad, sos_eos, mask, params=params)
    
    def midi_to_tokens(self, midi: MidiFile, *args, **kwargs) -> List[List[Union[int, List[int]]]]:
        return self.structured_interval_tokenizer.midi_to_tokens(midi, *args, **kwargs)

    def _create_vocabulary(self, sos_eos_tokens: bool = None) -> Vocabulary:
        vocab = self.structured_interval_tokenizer.vocab
        vocab.remove_token_type('Velocity')
        return vocab
    
    @staticmethod
    def break_simultaneous_tokens(next_token, base_tokenizer_vocab):
        next_token_type = base_tokenizer_vocab.token_type(next_token)
        return next_token_type == 'Time-Shift' and \
            base_tokenizer_vocab.token_to_event[next_token] != "Time-Shift_0.0.1"
            
    @staticmethod
    def preprocess_track_tokens(track_tokens, base_tokenizer_vocab):
        # Add a dummy Time-Shift token to trigger the simulateneous tokens
        if base_tokenizer_vocab.token_type(track_tokens[0]) != 'Time-Shift': # If the first token is not already a rest
            track_tokens.insert(0, base_tokenizer_vocab.event_to_token["Time-Shift_0.0.1"])
        return track_tokens

    @staticmethod
    def postprocess_track_tokens(track_tokens, self_vocab):
        # Remove the first dummy Time-Shift token
        if self_vocab.token_to_event[track_tokens[0]] == "Time-Shift_0.0.1": # If the first token wasn't already a rest
            track_tokens.pop(0)
        return track_tokens
    
    @staticmethod
    def skip_appending_next_token(next_token, base_tokenizer_vocab):
        next_token_type = base_tokenizer_vocab.token_type(next_token)
        return next_token_type == 'Time-Shift'
    
    @staticmethod
    def postprocess_simultaneous_tokens(simultaneous_tokens, self_vocab):
        temp_simultaneous_tokens = []
        TS_0_token = self_vocab.event_to_token["Time-Shift_0.0.1"]
        for token in simultaneous_tokens:
            token_type = self_vocab.token_type(token)
            if token_type == 'VPitchShift':
                temp_simultaneous_tokens += [TS_0_token, token]
            else:
                temp_simultaneous_tokens.append(token)
        return temp_simultaneous_tokens
    
    def tokens_to_midi(self, tokens: List[List[Union[int, List[int]]]],
            programs: Optional[List[Tuple[int, bool]]] = None, output_path: Optional[str] = None,
            time_division: Optional[int] = TIME_DIVISION) -> MidiFile:
        return self.structured_interval_tokenizer.tokens_to_midi(tokens, programs, output_path, time_division)
    
    
    def add_velocities(self, track_tokens, base_tokenizer_vocab):
        for token in base_tokenizer_vocab._token_to_event:
            frmt_token = base_tokenizer_vocab._token_to_event[token]
            if frmt_token == 'Velocity_127':
                velocity_token = token
        new_tokens = []
        for token in track_tokens:
            if base_tokenizer_vocab.token_type(token) == 'Pitch':
                new_tokens += [token, velocity_token]
            else:
                new_tokens.append(token)
        return new_tokens



# ======================================================================================================


class TSDVelocityMute(TSD):
    def __init__(
        self,
        pitch_range: range = PITCH_RANGE,
        beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
        nb_velocities: int = NB_VELOCITIES,
        additional_tokens: Dict[str, Union[bool, int]] = ADDITIONAL_TOKENS,
        pad: bool = True,
        sos_eos: bool = False,
        mask: bool = False,
        params=None,
        *args, **kwargs,
    ):
        # No additional tokens
        additional_tokens["Chord"] = False  # Incompatible additional token
        additional_tokens["Rest"] = False
        additional_tokens["Tempo"] = False
        additional_tokens["TimeSignature"] = False
        super().__init__(
            pitch_range, beat_res,
            1, # nb_velocities
            additional_tokens, pad, sos_eos, mask, params=params,
        )

    def track_to_tokens(self, track: Instrument) -> List[int]:
        tsd_tokens = super().track_to_tokens(track)
        new_tokens = []
        for token in tsd_tokens:
            if self.vocab.token_type(token) != "Velocity":
                new_tokens.append(token)
        return new_tokens

    def _create_vocabulary(self, sos_eos_tokens: bool = None) -> Vocabulary:
        vocab = super()._create_vocabulary(sos_eos_tokens=sos_eos_tokens)
        return vocab
    
    
# ===================================================================================================

class TSDIntervalVelocityMute(TSD):
    def __init__(
        self,
        include_first_note: bool = False,
        pitch_range: range = PITCH_RANGE,
        beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
        nb_velocities: int = NB_VELOCITIES,
        additional_tokens: Dict[str, Union[bool, int]] = ADDITIONAL_TOKENS,
        pad: bool = True,
        sos_eos: bool = False,
        mask: bool = False,
        params=None,
    ):
        # No additional tokens
        additional_tokens["Chord"] = False  # Incompatible additional token
        additional_tokens["Rest"] = False
        additional_tokens["Tempo"] = False
        additional_tokens["TimeSignature"] = False
        self.tsd_interval_tokenizer = interval_tokenizer(
            TSDVelocityMute,
            break_simultaneous_tokens=self.break_simultaneous_tokens,
            preprocess_track_tokens=self.preprocess_track_tokens,
            postprocess_track_tokens=self.postprocess_track_tokens,
            skip_appending_next_token=self.skip_appending_next_token,
            # postprocess_simultaneous_tokens=self.postprocess_simultaneous_tokens,
            include_first_note=include_first_note,
            pitch_range=pitch_range,
            beat_res=beat_res,
            nb_velocities=1,
            additional_tokens=additional_tokens,
            pad=pad,
            sos_eos=sos_eos,
            mask=mask,
            params=params,
        )
        
        super().__init__(pitch_range, beat_res, 1, additional_tokens, pad, sos_eos, mask, params=params)
    
    def midi_to_tokens(self, midi: MidiFile, *args, **kwargs) -> List[List[Union[int, List[int]]]]:
        return self.tsd_interval_tokenizer.midi_to_tokens(midi, *args, **kwargs)

    def _create_vocabulary(self, sos_eos_tokens: bool = None) -> Vocabulary:
        vocab = self.tsd_interval_tokenizer.vocab
        vocab.remove_token_type('Velocity')
        return vocab
    
    @staticmethod
    def break_simultaneous_tokens(next_token, base_tokenizer_vocab):
        next_token_type = base_tokenizer_vocab.token_type(next_token)
        return next_token_type in ['Time-Shift', 'PAD'] and \
            base_tokenizer_vocab.token_to_event[next_token] != "PAD_None"
            
    @staticmethod
    def preprocess_track_tokens(track_tokens, base_tokenizer_vocab):
        # Add a dummy Time-Shift token to trigger the simulateneous tokens
        if base_tokenizer_vocab.token_type(track_tokens[0]) != 'Time-Shift': # If the first token is not already a rest
            track_tokens.insert(0, base_tokenizer_vocab.event_to_token["PAD_None"])
        return track_tokens

    @staticmethod
    def postprocess_track_tokens(track_tokens, self_vocab):
        # Remove the first dummy Time-Shift token
        if self_vocab.token_to_event[track_tokens[0]] == "PAD_None": # If the first token wasn't already a rest
            track_tokens.pop(0)
        return track_tokens
    
    @staticmethod
    def skip_appending_next_token(next_token, base_tokenizer_vocab):
        next_token_type = base_tokenizer_vocab.token_type(next_token)
        return next_token_type == 'Time-Shift'
    
    def tokens_to_midi(self, tokens: List[List[Union[int, List[int]]]],
            programs: Optional[List[Tuple[int, bool]]] = None, output_path: Optional[str] = None,
            time_division: Optional[int] = TIME_DIVISION) -> MidiFile:
        return self.tsd_interval_tokenizer.tokens_to_midi(tokens, programs, output_path, time_division)
            


class PitchOnly(REMI):
    def __init__(
        self,
        pitch_range: range = PITCH_RANGE,
        beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
        nb_velocities: int = NB_VELOCITIES,
        additional_tokens: Dict[str, Union[bool, int]] = ADDITIONAL_TOKENS,
        pad: bool = True,
        sos_eos: bool = False,
        mask: bool = False,
        params=None,
        *args, **kwargs,
    ):
        # No additional tokens
        additional_tokens["Chord"] = False  # Incompatible additional token
        additional_tokens["Rest"] = False
        additional_tokens["Tempo"] = False
        additional_tokens["TimeSignature"] = False
        super().__init__(
            pitch_range, beat_res,
            1,  # nb_velocities
            additional_tokens, pad, sos_eos, mask, params=params,
        )

    def track_to_tokens(self, track: Instrument) -> List[int]:
        events = []
        for note in track.notes:
            events.append(Event(type_='Pitch', value=note.pitch, time=note.start, desc=note.pitch))
        return self.events_to_tokens(events)

    def _create_vocabulary(self, sos_eos_tokens: bool = None) -> Vocabulary:
        # vocab = super()._create_vocabulary(sos_eos_tokens=sos_eos_tokens)
        vocab = Vocabulary(pad=self._pad, sos_eos=self._sos_eos, mask=self._mask)
        vocab.add_event(f'Pitch_{i}' for i in self.pitch_range)
        return vocab


# ======================================================================================================
# ====================================== TESTS =========================================================
# ======================================================================================================


from .utils import *

def tests_REMIVelocityMute():
    midi_file = "./corpus/folk_norm/china/china01.mid"
    midi = MidiFile(midi_file)
    tokenizer = REMIVelocityMute()
    tokens = tokenizer.midi_to_tokens(midi)[0]
    
    print("Total length:", len(tokens))
    print("--------------")
    print_tokens(tokenizer, tokens)
    # print_vocab(tokenizer)
    
    print()
    tokenizer_remi = REMI()
    tokens_remi = tokenizer_remi.midi_to_tokens(midi)[0]
    
    print("Total length:", len(tokens_remi))
    print("--------------")
    print_tokens(tokenizer_remi, tokens_remi)
    # print_vocab(tokenizer)
    
    
def tests_interval_tokenizer():
    # ===== Tests intervcal_tokenizer =====
    # midi_file = "./corpus/folk_norm/china/china01.mid"
    # midi_file = "./corpus/piano_norm/haydn/sonata12-1.mid"
    # midi_file = "corpus/tavern/BOp76/Bo76_00_01_score.mid"
    # midi_file = "./data_dummy/lbd.mid"
    # midi_file = "./data_dummy/monophonic.mid"
    midi_file = "./data_dummy/polyphonic.mid"
    # midi_file = "./data_dummy/chords.mid"
    midi = MidiFile(midi_file)
    
    
    # tokenizer = REMI()
    tokenizer = Structured()
    
    absolute_tokens = tokenizer.midi_to_tokens(midi)[0]
    print_tokens(tokenizer, absolute_tokens)
    print()
    
    # tokenizer = REMIInterval(include_first_note=False)
    # tokenizer = REMIInterval(include_first_note=True)
    # tokenizer = REMIIntervalVelocityMute(include_first_note=False)
    # tokenizer = REMIIntervalVelocityMute(include_first_note=True)
    # tokenizer = StructuredInterval(include_first_note=False)
    # tokenizer = StructuredInterval(include_first_note=True)
    # tokenizer = StructuredIntervalVelocityMute(include_first_note=False)
    # tokenizer = StructuredIntervalVelocityMute(include_first_note=True)
    # tokenizer = TSDVelocityMute()
    # tokenizer = TSDIntervalVelocityMute(include_first_note=False)
    tokenizer = PitchOnly()
    print(tokenizer.vocab)
    # print_vocab(tokenizer)
    
    tokens = tokenizer.midi_to_tokens(midi)[0]
    print_tokens(tokenizer, tokens)
    # print(tokens)
    
    # print()
    # midi = tokenizer.tokens_to_midi([tokens])
    # print(midi.instruments[0].notes)
    # midi.dump('reconstruct.mid')


    
if __name__ == "__main__":
    # tests_REMIVelocityMute()
    tests_interval_tokenizer()
    pass