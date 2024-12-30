from pathlib import Path

import pandas as pd

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)

    def log_predictions(
            self, text, log_probs, log_probs_length, audio_path, examples_to_log=10, beam_size=5, **batch
    ):
        """
        Log predictions using both greedy decoding and beam search.

        Args:
            text (List[str]): ground truth texts
            log_probs (torch.Tensor): [B, T, V] log probabilities from model
            log_probs_length (torch.Tensor): [B] valid length of each sequence
            audio_path (List[str]): paths to audio files
            examples_to_log (int): number of examples to log in the table
            beam_size (int): beam size for beam search decoding
        """
        # Get greedy decode results (argmax)
        argmax_inds = log_probs.cpu().argmax(-1).numpy()
        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(argmax_inds, log_probs_length.numpy())
        ]
        argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
        argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]

        # Beam search decoding
        beam_search_texts = []
        for i, length in enumerate(log_probs_length):
            sequence_log_probs = log_probs[i, :length]  # [T, V]

            # Initialize beam
            beam = [([], 0.0)]  # (sequence, log_prob)

            # For each timestep
            for t in range(length):
                candidates = []

                # Expand each beam
                for seq, score in beam:
                    probs = sequence_log_probs[t].cpu()  # [V]
                    top_probs, top_indices = probs.topk(beam_size)

                    for prob, idx in zip(top_probs, top_indices):
                        new_seq = seq + [idx.item()]
                        new_score = score + prob.item()
                        candidates.append((new_seq, new_score))

                # Select top beam_size candidates
                candidates.sort(key=lambda x: x[1], reverse=True)
                beam = candidates[:beam_size]

            # Get best sequence
            best_sequence = beam[0][0]
            beam_search_texts.append(self.text_encoder.ctc_decode(best_sequence))

        # Prepare rows for logging table
        rows = {}
        tuples = list(zip(argmax_texts, beam_search_texts, text, argmax_texts_raw, audio_path))

        for greedy_pred, beam_pred, target, raw_pred, audio_path in tuples[:examples_to_log]:
            target = self.text_encoder.normalize_text(target)
            greedy_wer = calc_wer(target, greedy_pred) * 100
            greedy_cer = calc_cer(target, greedy_pred) * 100
            beam_wer = calc_wer(target, beam_pred) * 100
            beam_cer = calc_cer(target, beam_pred) * 100

            rows[Path(audio_path).name] = {
                "target": target,
                "raw prediction": raw_pred,
                "greedy prediction": greedy_pred,
                "beam search prediction": beam_pred,
                "greedy WER": greedy_wer,
                "beam WER": beam_wer,
                "greedy CER": greedy_cer,
                "beam CER": beam_cer,
            }

        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )
