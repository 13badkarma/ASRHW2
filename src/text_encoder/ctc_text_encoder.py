import re
from string import ascii_lowercase
from typing import List, Optional, Union
from collections import defaultdict

import torch
import numpy as np

class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, language_model=None, **kwargs):
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
        CTC beam search decoding using log probabilities.
        
        Args:
            logits: Log probabilities from model [T, V]
            beam_size: Beam width
            lm_weight: Weight for language model scoring
            beam_prune_logp: Prune beams with log prob below this threshold
        Returns:
            text: Best decoded text according to beam search
        """
        # Ensure input is proper log probabilities
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits)
        if len(logits.shape) == 1:
            # Convert indices to one-hot log probs
            indices = logits
            logits = torch.zeros((len(indices), len(self.vocab)))
            logits[range(len(indices)), indices] = 0
            logits[range(len(indices)), :] = float('-inf')
            
        # Initialize beam with empty sequence
        beam = {('', self.EMPTY_TOK): 0.0}  # (prefix, last_char): log_prob
        
        T = logits.shape[0]  # sequence length
        
        for t in range(T):
            new_beam = defaultdict(lambda: float('-inf'))
            
            for (prefix, last_char), prefix_logp in beam.items():
                # For each vocabulary item
                for v in range(len(self.vocab)):
                    char = self.ind2char[v]
                    log_p = logits[t, v].item()
                    
                    # Skip if probability is too low
                    if log_p < beam_prune_logp:
                        continue
                        
                    # Apply CTC rules
                    if char == self.EMPTY_TOK:
                        new_prefix, new_last = prefix, last_char
                    elif char == last_char:
                        new_prefix, new_last = prefix, last_char
                    else:
                        new_prefix = prefix + char
                        new_last = char
                        
                    # Add language model score if available
                    lm_score = 0.0
                    if self.language_model is not None and new_prefix != prefix:
                        lm_score = lm_weight * self.language_model.score(new_prefix)
                    
                    # Update beam with log-sum-exp
                    new_logp = prefix_logp + log_p + lm_score
                    current = new_beam[(new_prefix, new_last)]
                    new_beam[(new_prefix, new_last)] = torch.logaddexp(
                        torch.tensor(current),
                        torch.tensor(new_logp)
                    ).item()
            
            # Prune beam
            beam_items = sorted(new_beam.items(), key=lambda x: x[1], reverse=True)
            beam = dict(beam_items[:beam_size])
            
            # If all beams are very improbable, restart with empty beam
            if all(p < beam_prune_logp for p in beam.values()):
                beam = {('', self.EMPTY_TOK): 0.0}
        
        # Handle empty result
        if not beam:
            return ""
            
        # Return highest probability sequence
        return max(beam.items(), key=lambda x: x[1])[0][0]

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
