import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torchaudio
import logging

@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def test_batch_with_train_augs(config):
    """
    Test model on a batch of audio files using training augmentations
    """
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Setup device
    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # Setup text encoder
    text_encoder = instantiate(config.text_encoder)
    
    # Initialize transforms
    get_spectrogram = instantiate(config.transforms.instance_transforms.train.get_spectrogram)
    audio_transforms = instantiate(config.transforms.instance_transforms.train.audio_transforms)
    spectrogram_transforms = instantiate(config.transforms.instance_transforms.train.spectrogram)
    
    # Список путей к файлам
    audio_paths = [
        '/home/pavelpers/Jupyter/ASRHW2/data/datasets/librispeech/train-clean-100/2843/152918/2843-152918-0033.flac',
        '/home/pavelpers/Jupyter/ASRHW2/data/datasets/librispeech/train-clean-100/248/130644/248-130644-0034.flac',
        '/home/pavelpers/Jupyter/ASRHW2/data/datasets/librispeech/train-clean-100/6272/70191/6272-70191-0007.flac',
        '/home/pavelpers/Jupyter/ASRHW2/data/datasets/librispeech/train-clean-100/6078/54013/6078-54013-0001.flac',
        '/home/pavelpers/Jupyter/ASRHW2/data/datasets/librispeech/train-clean-100/2764/36616/2764-36616-0017.flac',
        '/home/pavelpers/Jupyter/ASRHW2/data/datasets/librispeech/train-clean-100/458/126305/458-126305-0028.flac',
        '/home/pavelpers/Jupyter/ASRHW2/data/datasets/librispeech/train-clean-100/3214/167607/3214-167607-0007.flac',
        '/home/pavelpers/Jupyter/ASRHW2/data/datasets/librispeech/train-clean-100/4406/16883/4406-16883-0003.flac',
        '/home/pavelpers/Jupyter/ASRHW2/data/datasets/librispeech/train-clean-100/1116/137572/1116-137572-0041.flac',
        '/home/pavelpers/Jupyter/ASRHW2/data/datasets/librispeech/train-clean-100/3807/4923/3807-4923-0035.flac',
        '/home/pavelpers/Jupyter/ASRHW2/data/datasets/librispeech/train-clean-100/5463/39174/5463-39174-0009.flac',
        '/home/pavelpers/Jupyter/ASRHW2/data/datasets/librispeech/train-clean-100/3436/172162/3436-172162-0025.flac'
    ]
    
    # Prepare items for collate_fn
    dataset_items = []
    for path in audio_paths:
        waveform, sr = torchaudio.load(path)
        # Применяем аудио аугментации
        waveform = audio_transforms(waveform)
        
        # Получаем спектрограмму
        spectrogram = get_spectrogram(waveform).squeeze(1)
        # Применяем аугментации спектрограммы
        spectrogram = spectrogram_transforms(spectrogram)
        
        item = {
            "audio": waveform,
            "audio_path": path,
            "spectrogram": spectrogram,
            "text": "test",  # если есть транскрипции, можно добавить
            "text_encoded": torch.zeros(1, 1, dtype=torch.long)  # dummy tensor для коллейта
        }
        dataset_items.append(item)
    
    # Collate items
    batch = collate_fn(dataset_items)
    
    # Move relevant tensors to device
    batch["spectrogram"] = batch["spectrogram"].to(device)
    batch["spectrogram_length"] = batch["spectrogram_length"].to(device)
    
    # Build model using config
    model = instantiate(
        config.model,
        n_tokens=len(text_encoder)
    ).to(device)
    
    # Load weights if specified in config
    if "checkpoint_path" in config:
        logger.info(f"Loading checkpoint from {config.checkpoint_path}")
        checkpoint = torch.load(config.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
    
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        output = model(spectrogram=batch["spectrogram"],
                      spectrogram_length=batch["spectrogram_length"])
    
    # Decode output for each sample in batch
    log_probs = output["log_probs"]
    decoded_outputs = []
    for i in range(len(audio_paths)):
        decoded_text = text_encoder.decode(log_probs[i].cpu().numpy())
        decoded_outputs.append(decoded_text)
    
    # Print results
    logger.info(f"Batch size: {len(audio_paths)}")
    logger.info(f"Input spectrogram shape: {batch['spectrogram'].shape}")
    logger.info(f"Output log probs shape: {output['log_probs'].shape}")
    logger.info("Decoded texts:")
    for i, (path, text) in enumerate(zip(audio_paths, decoded_outputs)):
        logger.info(f"{i+1}. {path.split('/')[-1]}: {text}")
    
    return output, decoded_outputs

def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    """
    # Get batch size
    batch_size = len(dataset_items)

    # Find max lengths for padding
    max_audio_len = max(item["audio"].shape[1] for item in dataset_items)
    max_spectrogram_len = max(item["spectrogram"].shape[2] for item in dataset_items)
    max_text_encoded_len = max(item["text_encoded"].shape[1] for item in dataset_items)

    # Initialize tensors for the batch
    audio_batch = torch.zeros(batch_size, 1, max_audio_len)
    spectrogram_batch = torch.zeros(
        batch_size,
        dataset_items[0]["spectrogram"].shape[1],  # frequency dimension
        max_spectrogram_len
    )
    text_encoded_batch = torch.zeros(batch_size, max_text_encoded_len, dtype=torch.long)

    # Keep track of actual lengths
    audio_lengths = []
    spectrogram_lengths = []
    text_encoded_lengths = []
    texts = []
    audio_paths = []

    # Fill in the batches
    for i, item in enumerate(dataset_items):
        # Audio
        audio = item["audio"]
        audio_len = audio.shape[1]
        audio_batch[i, :, :audio_len] = audio
        audio_lengths.append(audio_len)

        # Spectrogram
        spectrogram = item["spectrogram"]
        spec_len = spectrogram.shape[2]
        spectrogram_batch[i, :, :spec_len] = spectrogram
        spectrogram_lengths.append(spec_len)

        # Text
        text_encoded = item["text_encoded"]
        text_len = text_encoded.shape[1]
        text_encoded_batch[i, :text_len] = text_encoded
        text_encoded_lengths.append(text_len)

        # Original text and paths
        texts.append(item["text"])
        audio_paths.append(item["audio_path"])

    # Convert lengths to tensors
    audio_lengths = torch.tensor(audio_lengths)
    spectrogram_lengths = torch.tensor(spectrogram_lengths)
    text_encoded_lengths = torch.tensor(text_encoded_lengths)

    return {
        "audio": audio_batch,
        "audio_length": audio_lengths,
        "spectrogram": spectrogram_batch,
        "spectrogram_length": spectrogram_lengths,
        "text_encoded": text_encoded_batch,
        "text_encoded_length": text_encoded_lengths,
        "text": texts,
        "audio_path": audio_paths,
    }

if __name__ == "__main__":
    test_batch_with_train_augs()