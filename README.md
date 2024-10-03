# Analyzing Byte-Pair Encoding on Monophonic and Polyphonic Symbolic Music: A Focus on Musical Phrase Segmentation

*Accepted to 3rd Workshop on NLP for Music and Audio (NLP4MusA)*

*Dinh-Viet-Toan Le, Louis Bigo, Mikaela Keller*

**Abstract** -
Byte-Pair Encoding (BPE) is an algorithm commonly used in Natural Language Processing to build a vocabulary of subwords, which has been recently applied to symbolic music. Given that symbolic music can differ significantly from text, particularly with polyphony, we investigate how BPE behaves with different types of musical content.
This study provides a qualitative analysis of BPE's behavior across various instrumentations and evaluates its impact on a musical phrase segmentation task for both monophonic and polyphonic music. Our findings show that the BPE training process is highly dependent on the instrumentation and that BPE ``supertokens'' succeed in capturing abstract musical content. In a musical phrase segmentation task, BPE notably improves performance in a polyphonic setting, but enhances performance in monophonic tunes only within a specific range of BPE merges.

[![arXiv Badge](https://img.shields.io/badge/arXiv-2410.01448-B31B1B?logo=arxiv&logoColor=fff&style=flat)](
http://arxiv.org/abs/2410.01448)

---

## Setup

Create environment:
```
conda create -n envtokenization python=3.9.2
conda activate envtokenization
```

Install requirements:
```
pip install --no-deps -r requirements.txt
```

## Reproduce figures from pre-computed data
1. Download pre-computed data and models [here](XXX)
2. Data content:
- `exp231_save` and `exp232_save`: Pretrained models for phrase segmentation (Figure 3). To put in your preferred directory (modify `model_path_231` and `model_path_232` for phrase segmentation performances in `figures_paper.ipynb`).
- `corpus`: raw MIDI files, phrase segmentation annotations. To put in `./corpus`.
- `bpe_tokenizers`: pre-trained BPE tokenizers. To put in `./bpe_tokenizers`.
- `results`: pre-computed data for text-BPE vs. music-BPE (Figure 1) and supertokens with pitches (Figure 4). To put in `./results`.
3. Run notebook `figures_paper.ipynb` 

## Training

### BPE tokenizer
Use:
```
python train_bpe.py --corpus=<poly|mono> --n_merges=<int> 
```

Outputs:
- `train.bpe`: Pre-trained BPE tokenizer
- `train.bpe.frq`: Supertoken frequency (for Figure 1) 


Options:
- `--tokenizer_init=<InitialTokenizer>`: start from an already trained BPE tokenizer. If none, start from the initial vocabulary.
- `--output_file=<OutputFilename>` : output filename (default: `train.bpe`)
- `--bypass_tokenize` : bypass the tokenization step before BPE (i.e. the path `'data_tokenized/{TokenizerName}/{Corpus}/train'` already exists), because it can be very long for several trainings... Warning if used with `--tokenizer_init`, make sure that it has been tokenized with THIS initial tokenizer.

### Musical phrase detection

#### Monophonic
Use:
```
python exp231_clfdata_tf.py --config=<config_file>
```

Required:
- Pre-trained BPE tokenizer (according to the `bpe_savepath` field in the config file.)

Outputs:
- Trained models saved at `PATH_CKPT` (defined in `exp231_clfdata_tf.py`)

Options:
- `--precompute_data`: builds pre-computed data `mtc_clfdata_<TokenizerName>_bpe<NumBPE>.feather`
- `--seed_split=<int>`

*No BPE*
```
python exp231_clfdata_tf.py --config=config/clfdata_transformers_withbpe.yaml
```

*With BPE*
```
python exp231_clfdata_tf.py --config=config/clfdata_transformers_withbpe.yaml
```


#### Polyphonic

Use:
```
python exp232_clfdata_tf.py --config=<config_file>
```

Required:
- Pre-trained BPE tokenizer (according to the `bpe_savepath` field in the config file.)

Outputs:
- Trained models saved at `PATH_CKPT` (defined in `exp232_clfdata_tf.py`)

Options:
- `--precompute_data`: builds pre-computed data `mtc_piano_clfdata_<TokenizerName>_bpe<NumBPE>_chunkafter.feather`
- `--seed_split=<int>`

*No BPE*
```
python exp232_clfdata_tf.py --config=config/clfdata_piano_transformers_nobpe.yaml
```

*With BPE*
```
python exp232_clfdata_tf.py --config=config/clfdata_piano_transformers_withbpe.yaml
```


## Evaluation

Required:
- Trained models for phrase segmentation. In particular, only the `best_loss.pt` model (trained by `exp231_clfdata_tf.py` or `exp232_clfdata_tf.py`) is evaluated.
- Pre-computed data with the correct number of BPE merges (ex: `mtc_clfdata_REMIVelocityMute_bpe4096.feather`) in the current folder.

Outputs:
- `perfo.json` created in the checkpoint folder.

The same scripts can be used to evaluate both BPE and non-BPE models.

### Monophonic
```
python exp231_clfdata_evaluate.py <CheckpointFolder> (cuda:<device_number> optional)
```

### Polyphonic
```
python exp232_clfdata_evaluate.py <CheckpointFolder> (cuda:<device_number> optional)
```

## Citation BibTex
If you find this work helpful and use our code in your research, please cite our paper:
```
@inproceedings{le2024analyzing,
  title={Analyzing Byte-Pair Encoding on Monophonic and Polyphonic Symbolic Music: A Focus on Musical Phrase Segmentation},
  author={Le, Dinh-Viet-Toan and Bigo, Louis and Keller, Mikaela},
  booktitle={Proceedings of the 3rd Workshop on NLP for Music and Audio (NLP4MusA)},
  year={2024},
}
```