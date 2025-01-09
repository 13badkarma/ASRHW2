from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer


class DecodingMethod(Enum):
    GREEDY = "greedy"
    BEAM = "beam"
    BEAM_LM = "beam_lm"


@dataclass
class DecodingConfig:
    method: DecodingMethod = DecodingMethod.GREEDY
    beam_size: int = 20
    lm_weight: float = 0.5
    beam_prune_logp: float = -15.0


class CERMetric(BaseMetric):
    """
    Character Error Rate metric with support for different decoding methods:
    - Greedy decoding
    - Beam search
    - Beam search with language model
    """

    def __init__(
            self,
            text_encoder,
            decoding_cfg: Optional[DecodingConfig] = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.decoding_cfg = decoding_cfg or DecodingConfig()

        # Custom name based on decoding method
        self.name = f"cer_{self.decoding_cfg.method.value}"

    def __call__(
            self,
            log_probs: Tensor,
            log_probs_length: Tensor,
            text: List[str],
            **kwargs
    ):
        """
        Calculate CER using specified decoding method.

        Args:
            log_probs: Log probabilities from model [B, T, V]
            log_probs_length: Valid length for each sequence [B]
            text: Ground truth texts
        Returns:
            Average CER across batch
        """
        cers = []
        predictions = log_probs.cpu()
        lengths = log_probs_length.cpu()

        for pred, length, target in zip(predictions, lengths, text):
            target = self.text_encoder.normalize_text(target)

            # Get predictions based on decoding method
            if self.decoding_cfg.method == DecodingMethod.GREEDY:
                pred_indices = torch.argmax(pred[:length], dim=-1).numpy()
                pred_text = self.text_encoder.ctc_decode(pred_indices)

            elif self.decoding_cfg.method == DecodingMethod.BEAM:
                pred_text = self.text_encoder.ctc_decode(
                    pred[:length],
                    beam_size=self.decoding_cfg.beam_size
                )

            elif self.decoding_cfg.method == DecodingMethod.BEAM_LM:
                # Use text_encoder's beam search with LM scoring
                pred_text = self.text_encoder.ctc_decode(
                    pred[:length],
                    beam_size=self.decoding_cfg.beam_size,
                    lm_weight=self.decoding_cfg.lm_weight,
                    beam_prune_logp=self.decoding_cfg.beam_prune_logp
                )

            cers.append(calc_cer(target, pred_text))

        return sum(cers) / len(cers)


# Predefined metrics for different decoding methods
class ArgmaxCERMetric(CERMetric):
    """Greedy CER metric for backward compatibility"""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            decoding_cfg=DecodingConfig(method=DecodingMethod.GREEDY),
            **kwargs
        )


class BeamSearchCERMetric(CERMetric):
    """Beam Search CER metric"""

    def __init__(self, *args, beam_size: int = 5, **kwargs):
        super().__init__(
            *args,
            decoding_cfg=DecodingConfig(
                method=DecodingMethod.BEAM,
                beam_size=beam_size
            ),
            **kwargs
        )


class BeamSearchLMCERMetric(CERMetric):
    """Beam Search with LM CER metric"""

    def __init__(
            self,
            *args,
            beam_size: int = 20,
            lm_weight: float = 0.3,
            beam_prune_logp: float = -8.0,
            **kwargs
    ):
        super().__init__(
            *args,
            decoding_cfg=DecodingConfig(
                method=DecodingMethod.BEAM_LM,
                beam_size=beam_size,
                lm_weight=lm_weight,
                beam_prune_logp=beam_prune_logp
            ),
            **kwargs
        )
