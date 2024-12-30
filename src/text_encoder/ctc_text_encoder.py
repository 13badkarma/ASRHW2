import re
from string import ascii_lowercase
from typing import List, Optional
from collections import defaultdict

import torch
import numpy as np

class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, language_model=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
            language_model: Optional language model for beam search scoring
        """
        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.language_model = language_model

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

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
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds: List[int], beam_size: Optional[int] = None) -> str:
        """
        CTC decoding with optional beam search and language model scoring.

        Args:
            inds (List[int]): List of token indices from model output
            beam_size (Optional[int]): If provided, use beam search with given width
        Returns:
            text (str): Decoded text with CTC collapsing rules applied
        """
        if beam_size is not None:
            return self._ctc_beam_search(inds, beam_size)

        # Basic greedy CTC decoding
        processed = []
        prev_token = None

        for ind in inds:
            token = self.ind2char[int(ind)]

            # Rule 1: Skip empty tokens
            if token == self.EMPTY_TOK:
                prev_token = token
                continue

            # Rule 2: Skip repeated tokens
            if token == prev_token:
                continue

            processed.append(token)
            prev_token = token

        return "".join(processed).strip()

    def _ctc_beam_search(self, inds: List[int], beam_size: int) -> str:
        """
        CTC beam search decoding with optional language model scoring.

        Args:
            inds (List[int]): Token indices from model output
            beam_size (int): Beam width
        Returns:
            text (str): Best decoded text according to beam search
        """
        # Initialize beam with empty sequence
        beam = [([], 0.0)]  # (sequence, log_probability)

        for t, ind in enumerate(inds):
            candidates = defaultdict(float)

            for seq, score in beam:
                token = self.ind2char[int(ind)]

                # Skip empty token case
                if token == self.EMPTY_TOK:
                    candidates[tuple(seq)] = max(candidates[tuple(seq)], score)
                    continue

                # Skip repeated token case
                if seq and token == seq[-1]:
                    candidates[tuple(seq)] = max(candidates[tuple(seq)], score)
                    continue

                # Add new token to sequence
                new_seq = list(seq) + [token]
                new_score = score

                # Add language model score if available
                if self.language_model is not None:
                    new_score += self.language_model.score("".join(new_seq))

                candidates[tuple(new_seq)] = max(candidates[tuple(new_seq)], new_score)

            # Select top-k candidates
            beam = []
            for seq, score in sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:beam_size]:
                beam.append((list(seq), score))

        # Return best sequence
        best_seq = max(beam, key=lambda x: x[1])[0]
        return "".join(best_seq).strip()

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text