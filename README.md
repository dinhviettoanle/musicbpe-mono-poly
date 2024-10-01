# Analyzing Byte-Pair Encoding on Monophonic and Polyphonic Symbolic Music: A Focus on Musical Phrase Segmentation

*Dinh-Viet-Toan Le, Louis Bigo, Mikaela Keller*

**Abstract** -
Byte-Pair Encoding (BPE) is an algorithm commonly used in Natural Language Processing to build a vocabulary of subwords, which has been recently applied to symbolic music. Given that symbolic music can differ significantly from text, particularly with polyphony, we investigate how BPE behaves with different types of musical content.
This study provides a qualitative analysis of BPE's behavior across various instrumentations and evaluates its impact on a musical phrase segmentation task for both monophonic and polyphonic music. Our findings show that the BPE training process is highly dependent on the instrumentation and that BPE ``supertokens'' succeed in capturing abstract musical content. In a musical phrase segmentation task, BPE notably improves performance in a polyphonic setting, but enhances performance in monophonic tunes only within a specific range of BPE merges.



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

## Use pre-compute data
1. Download pre-computed data
2. Put data:
XXX
3. Run notebook `figures_paper.ipynb`

## Training

### BPE tokenizers
TODO


### Musical phrase detection
#### Monophonic
Use:
```
python exp231_clfdata_tf.py --config=<config_file>
```

Required:
- Pre-trained BPE tokenizer (according to the `bpe_savepath` field in the config file.)

Options:
- `--precompute_data`: builds pre-computed data `mtc_clfdata_<TokenizerName>_bpe<NumBPE>.feather`
- `--seed_split=<int>`

**No BPE**
```
python exp231_clfdata_tf.py --config=config/clfdata_transformers_withbpe.yaml
```

**With BPE**
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

Options:
- `--precompute_data`: builds pre-computed data `mtc_piano_clfdata_<TokenizerName>_bpe<NumBPE>_chunkafter.feather`
- `--seed_split=<int>`

**No BPE**
```
python exp231_clfdata_tf.py --config=config/clfdata_piano_transformers_nobpe.yaml
```

**With BPE**
```
python exp232_clfdata_tf.py --config=config/clfdata_piano_transformers_withbpe.yaml
```


## Evaluation
TODO