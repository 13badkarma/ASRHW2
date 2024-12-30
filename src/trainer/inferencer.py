import torch
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
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
        batch_size = batch["logits"].shape[0]  # [B, T, num_classes]
        current_id = batch_idx * batch_size

        for i in range(batch_size):
            # Get logits and decode them to text
            logits = batch["logits"][i].detach().cpu()  # [T, num_classes]
            predicted_indices = torch.argmax(logits, dim=-1)  # [T]
            predicted_text = self.text_encoder.decode(predicted_indices)

            # Get ground truth if available
            ground_truth = None
            if "text" in batch:
                ground_truth = batch["text"][i]

            output_id = current_id + i

            # Prepare output dictionary
            output = {
                "predicted_text": predicted_text,
                "ground_truth": ground_truth,
                "logits": logits,
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

        return self.evaluation_metrics.result()
