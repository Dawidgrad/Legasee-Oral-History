'''
Alteration to code from pyctcdecode's github build_ctcdecoder method to allow for easier use multiple language models
'''

import kenlm
from pyctcdecode.decoder import BeamSearchDecoderCTC
from pyctcdecode.alphabet import verify_alphabet_coverage, Alphabet, BPE_TOKEN
from typing import List, Optional, Collection

from pyctcdecode.language_model import AbstractLanguageModel, LanguageModel, MultiLanguageModel, load_unigram_set_from_arpa

import logging
from pyctcdecode.constants import (
    DEFAULT_ALPHA,
    DEFAULT_BEAM_WIDTH,
    DEFAULT_BETA,
    DEFAULT_HOTWORD_WEIGHT,
    DEFAULT_MIN_TOKEN_LOGP,
    DEFAULT_PRUNE_BEAMS,
    DEFAULT_PRUNE_LOGP,
    DEFAULT_SCORE_LM_BOUNDARY,
    DEFAULT_UNK_LOGP_OFFSET,
    MIN_TOKEN_CLIP_P,
)

logger = logging.getLogger(__name__)


def build_ctcdecoder(
    labels: List[str],
    kenlm_model_path: Optional[List[str]] = None,
    unigrams: Optional[Collection[str]] = None,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    unk_score_offset: float = DEFAULT_UNK_LOGP_OFFSET,
    lm_score_boundary: bool = DEFAULT_SCORE_LM_BOUNDARY,
) -> BeamSearchDecoderCTC:
    """Build a BeamSearchDecoderCTC instance with main functionality.
    Args:
        labels: class containing the labels for input logit matrices
        kenlm_model_path: path to kenlm n-gram language model [List] or None LMs must share same vocab!
        unigrams: list of known word unigrams
        alpha: weight for language model during shallow fusion
        beta: weight for length score adjustment of during scoring
        unk_score_offset: amount of log score offset for unknown tokens
        lm_score_boundary: whether to have kenlm respect boundaries when scoring
    Returns:
        instance of BeamSearchDecoderCTC
    """
    kenlm_model = None if kenlm_model_path is None else [kenlm.Model(el) for el in kenlm_model_path]

    if kenlm_model_path is not None and any(el.endswith('.arpa') for el in kenlm_model_path):
        logger.info("Using arpa instead of binary LM file, decoder instantiation might be slow.")
        
    if unigrams is None and kenlm_model_path is not None:
        if any(el.endswith('.arpa') for el in kenlm_model_path):
            unigrams = [load_unigram_set_from_arpa(el) for el in kenlm_model_path]
        else:
            logger.warning(
                "Unigrams not provided and cannot be automatically determined from LM file (only "
                "arpa format). Decoding accuracy might be reduced."
            )

    alphabet = Alphabet.build_alphabet(labels)
    if unigrams is not None:
        [verify_alphabet_coverage(alphabet, el) for el in unigrams]

    if kenlm_model is not None:
        language_models: Optional[List[AbstractLanguageModel]] = [LanguageModel(
            model,
            unigrams[i],
            alpha=alpha,
            beta=beta,
            unk_score_offset=unk_score_offset,
            score_boundary=lm_score_boundary,
        ) for i, model in enumerate(kenlm_model)]
        MultiLM = MultiLanguageModel(language_models)
    else:
        language_model = None
    return BeamSearchDecoderCTC(alphabet, MultiLM)


vocab_labels = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, "E": 5, "T": 6, "O": 7, "A": 8, "I": 9, "N": 10, "H": 11, "S": 12, "R": 13, "L": 14, "D": 15, "U": 16, "Y": 17, "W": 18, "M": 19, "C": 20, "G": 21, "F": 22, "P": 23, "B": 24, "K": 25, "'": 26, "V": 27, "J": 28, "X": 29, "Q": 30, "Z": 31}
word_delimiter_token = "|"

def get_vocab():
    vocab_dict = vocab_labels
    sort_vocab = sorted((value, key) for (key,value) in vocab_dict.items())
    vocab = []
    for _, key in sort_vocab:
        vocab.append(key.lower())
    vocab[vocab.index(word_delimiter_token)] = ' '
    return vocab