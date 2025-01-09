import torch
from tqdm.auto import tqdm
import pandas as pd
from src.metrics.tracker import MetricTracker
from src.logger.utils import plot_spectrogram
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer



class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        text_encoder,
        writer,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            text_encoder (CTCTextEncoder): text encoder.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        self.text_encoder = text_encoder
        self.writer = writer

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Process a batch for ASR inference, computing metrics and saving predictions.

        Args:
            batch_idx (int): the index of the current batch
            batch (dict): dict containing audio features and text labels
            metrics (MetricTracker): tracks metrics during inference
            part (str): partition name (e.g., 'val', 'test')

        Returns:
            batch (dict): original batch with added model outputs
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        # Get model predictions
        outputs = self.model(**batch)
        batch.update(outputs)

        # Calculate metrics if provided
        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        # Process and save predictions
        batch_size = batch["log_probs"].shape[0]  # [B, T, num_classes]
        current_id = batch_idx * batch_size

        for i in range(batch_size):
            # Get log_probs and lengths for decoding
            log_probs = batch["log_probs"][i].detach().cpu()  # [T, num_classes]
            log_probs_length = batch["log_probs_length"][i].item() if "log_probs_length" in batch else log_probs.shape[0]
            
            # Decode using CTCTextEncoder with configured decoding method
            predicted_text = self.text_encoder.ctc_decode(
                log_probs[:log_probs_length],
                beam_size=self.config.text_encoder.get("beam_size", None),  # If beam_size is set in config
                lm_weight=self.config.text_encoder.get("lm_weight", 0.3),  # If using language model
                beam_prune_logp=self.config.text_encoder.get("beam_prune_logp", -10.0)  # Beam pruning threshold
            ).lower()

            # Get ground truth if available
            ground_truth = None
            if "text" in batch:
                ground_truth = batch["text"][i]

            output_id = current_id + i

            # Prepare output dictionary
            output = {
                "predicted_text": predicted_text,
                "ground_truth": ground_truth,
                "log_probs": log_probs,
            }

            # Add spectrograms/features if needed
            if "spectrogram" in batch:
                output["spectrogram"] = batch["spectrogram"][i].detach().cpu()

            # Save predictions if save_path is specified
            if self.save_path is not None:
                torch.save(output, self.save_path / part / f"output_{output_id}.pth")

        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """
        self.is_train = False
        self.model.eval()

        self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        last_batch = None  # Store the last batch for logging
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )
                last_batch = batch  # Save reference to the last processed batch

            # Log the final batch with accumulated metrics
            if last_batch is not None:
                self._log_batch(
                    batch_idx=len(dataloader)-1, 
                    batch=last_batch,
                    part=part
                )

            final_metrics = self.evaluation_metrics.result()
            
            # Преобразуем метрики в DataFrame
            metrics_df = pd.DataFrame.from_dict(
                {part: final_metrics},  # используем part как ключ
                orient='index'
            )
            
            # Записываем таблицу с метриками
            if self.writer is not None:
                self.writer.add_table(
                    f"{part}/final_metrics",
                    metrics_df
                )
            
            return final_metrics

    def _log_batch(self, batch_idx, batch, part="inference"):
        """
        Log batch results including WER/CER metrics and predictions.

        Args:
            batch_idx (int): index of the current batch
            batch (dict): processed batch with model outputs
            part (str): partition name (e.g., 'test', 'val')
        """
        if self.writer is None:
            return

        # Log spectrograms if available
        if "spectrogram" in batch:
            self.log_spectrogram(batch["spectrogram"])

        # Log predictions and metrics
        if "log_probs" in batch and "text" in batch:
            self._log_predictions(
                batch_idx=batch_idx,
                text=batch["text"],
                log_probs=batch["log_probs"],
                log_probs_length=batch.get("log_probs_length"),
                part=part,
                audio_path=batch.get("audio_path", [f"audio_{i}" for i in range(len(batch["text"]))]),
            )

    def log_spectrogram(self, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)
        
        # Add spectrogram visualization
        self.writer.add_image("sample_spectrogram", spectrogram)

    def _log_predictions(
        self,
        batch_idx,
        text,
        log_probs,
        log_probs_length=None,
        part="inference",
        audio_path=None,
        examples_to_log=50,
    ):
        """
        Log detailed prediction results with metrics.

        Args:
            batch_idx (int): batch index
            text (List[str]): ground truth texts
            log_probs (Tensor): model predictions [B, T, V]
            log_probs_length (Tensor, optional): valid lengths for sequences
            part (str): dataset partition
            audio_path (List[str], optional): paths to audio files
            examples_to_log (int): number of examples to include in log
        """
        if audio_path is None:
            audio_path = [f"sample_{batch_idx}_{i}" for i in range(len(text))]

        # Get predictions using different decoding methods
        predictions = {}
        batch_size = log_probs.shape[0]
        
        for i in range(min(batch_size, examples_to_log)):
            seq_log_probs = log_probs[i]
            if log_probs_length is not None:
                seq_len = log_probs_length[i]
                seq_log_probs = seq_log_probs[:seq_len]
            
            # Greedy decoding
            greedy_text = self.text_encoder.ctc_decode(seq_log_probs).lower()
            
            # Beam search decoding
            beam_text = self.text_encoder.ctc_decode(
                seq_log_probs,
                beam_size=self.config.text_encoder.get("beam_size", 5),
                lm_weight=self.config.text_encoder.get("lm_weight", 0.0),
                beam_prune_logp=self.config.text_encoder.get("beam_prune_logp", -10.0)
            ).lower()
            
            # Calculate metrics
            target = self.text_encoder.normalize_text(text[i]).lower()
            
            predictions[audio_path[i]] = {
                "target": target,
                "greedy_decoded": greedy_text,
                "beam_decoded": beam_text,
                "wer_greedy": calc_wer(target, greedy_text) * 100,
                "cer_greedy": calc_cer(target, greedy_text) * 100,
                "wer_beam": calc_wer(target, beam_text) * 100,
                "cer_beam": calc_cer(target, beam_text) * 100
            }

        # Log predictions table

        if predictions:
            self.writer.add_table(
                f"{part}/predictions",
                pd.DataFrame.from_dict(predictions, orient="index")
            )