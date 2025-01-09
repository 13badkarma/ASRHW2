from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer

# TODO beam search / LM versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved

from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer


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


class WERMetric(BaseMetric):
    """
    Word Error Rate metric with support for different decoding methods:
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
        self.name = f"wer_{self.decoding_cfg.method.value}"

    def __call__(
            self,
            log_probs: Tensor,
            log_probs_length: Tensor,
            text: List[str],
            **kwargs
    ):
        """
        Calculate WER using specified decoding method.

        Args:
            log_probs: Log probabilities from model [B, T, V]
            log_probs_length: Valid length for each sequence [B]
            text: Ground truth texts
        Returns:
            Average WER across batch
        """
        wers = []
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
                pred_text = self.text_encoder.ctc_decode(
                    pred[:length],
                    beam_size=self.decoding_cfg.beam_size,
                    lm_weight=self.decoding_cfg.lm_weight,
                    beam_prune_logp=self.decoding_cfg.beam_prune_logp
                )

            wers.append(calc_wer(target, pred_text))

        return sum(wers) / len(wers)


# Predefined metrics for different decoding methods
class ArgmaxWERMetric(WERMetric):
    """Greedy WER metric for backward compatibility"""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            decoding_cfg=DecodingConfig(method=DecodingMethod.GREEDY),
            **kwargs
        )


class BeamSearchWERMetric(WERMetric):
    """Beam Search WER metric"""

    def __init__(self, *args, beam_size: int = 5, **kwargs):
        super().__init__(
            *args,
            decoding_cfg=DecodingConfig(
                method=DecodingMethod.BEAM,
                beam_size=beam_size
            ),
            **kwargs
        )


class BeamSearchLMWERMetric(WERMetric):
    """Beam Search with LM WER metric"""

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
