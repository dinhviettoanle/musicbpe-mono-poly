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
Run notebook `figures_paper.ipynb`

## Training
Coming soon...

## Evaluation
Coming soon...