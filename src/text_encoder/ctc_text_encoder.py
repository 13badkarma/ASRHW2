import re
from string import ascii_lowercase
from typing import List, Optional, Union, Tuple, Dict, Any
from collections import defaultdict

import torch
import numpy as np
import kenlm
from pyctcdecode.decoder import build_ctcdecoder
from pyctcdecode.constants import DEFAULT_BEAM_WIDTH, DEFAULT_PRUNE_LOGP, DEFAULT_MIN_TOKEN_LOGP

class LMTextNormalizer:
    """Класс для нормализации текста между CTC и языковой моделью"""
    @staticmethod
    def to_lm_format(text: str) -> str:
        """Преобразует текст в формат для языковой модели"""
        return text.upper()
    
    @staticmethod
    def from_lm_format(text: str) -> str:
        """Преобразует текст из формата языковой модели"""
        return text.lower()

class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, kenlm_path=None, **kwargs):
        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.text_normalizer = LMTextNormalizer()
        
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        
        # Initialize KenLM and pyctcdecode components
        self.kenlm_model = None
        self.decoder = None
        if kenlm_path:
            # Создаем словарь с учетом регистра для языковой модели
            lm_vocab = [self.text_normalizer.to_lm_format(c) for c in self.vocab]
            
            # Используем build_ctcdecoder с словарем в верхнем регистре
            self.decoder = build_ctcdecoder(
                labels=lm_vocab,  # Словарь в верхнем регистре для LM
                kenlm_model_path=kenlm_path
            )

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, 
                   logits: Union[torch.Tensor, np.ndarray, List[int]], 
                   beam_size: Optional[int] = None,
                   lm_weight: float = 0.0,
                   beam_prune_logp: float = -10.0) -> str:
        """
        CTC decoding with optional beam search and language model scoring.
        
        Args:
            logits: Either log probabilities [T, V] or token indices [T]
            beam_size: If provided, use beam search with given width
            lm_weight: Weight for language model scoring (if LM available)
            beam_prune_logp: Prune beams with log probability below this
        Returns:
            text: Decoded text after applying CTC rules
        """
        # Handle different input types
        if isinstance(logits, (list, np.ndarray)):
            if beam_size is None or len(logits.shape) == 1:
                return self._ctc_greedy_decode(logits)
            logits = torch.tensor(logits)
            
        if beam_size is None:
            return self._ctc_greedy_decode(torch.argmax(logits, dim=-1))
            
        return self._ctc_beam_search(logits, beam_size, lm_weight, beam_prune_logp)

    def _ctc_greedy_decode(self, indices: Union[torch.Tensor, np.ndarray, List[int]]) -> str:
        """Basic greedy CTC decoding"""
        # Convert to list of ints if needed
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()

        processed = []
        prev_token = None

        for ind in indices:
            token = self.ind2char[int(ind)]

            if token == self.EMPTY_TOK or token == prev_token:
                prev_token = token
                continue

            processed.append(token)
            prev_token = token

        return "".join(processed).strip()

    def _ctc_beam_search(self, 
                        logits: torch.Tensor,
                        beam_size: int,
                        lm_weight: float = 0.0,
                        beam_prune_logp: float = -10.0) -> str:
        """
        CTC beam search decoding using pyctcdecode's implementation.
        
        Args:
            logits: Log probabilities from model [T, V]
            beam_size: Beam width
            lm_weight: Weight for language model scoring
            beam_prune_logp: Prune beams with log prob below this threshold
        Returns:
            text: Best decoded text according to beam search
        """
        if self.decoder is None:
            raise ValueError("Decoder not initialized. Please provide kenlm_path during initialization.")
            
        # Convert logits to proper format for pyctcdecode
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()
        
        # If logits are indices, convert to one-hot log probs
        if len(logits.shape) == 1:
            indices = logits
            new_logits = np.full((len(indices), len(self.vocab)), float('-inf'))
            new_logits[range(len(indices)), indices] = 0
            logits = new_logits
        
        # Perform beam search decoding
        beams = self.decoder.decode_beams(
            logits,
            beam_width=beam_size,
            beam_prune_logp=beam_prune_logp,
            token_min_logp=beam_prune_logp
        )
        
        # Get the best beam and convert it back to lowercase
        if beams:
            return self.text_normalizer.from_lm_format(beams[0][0])
        return ""

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text