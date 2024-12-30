import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.

    Args:
        dataset_items: List of dicts from dataset.__getitem__
    Returns:
        result_batch: Dict with batched and padded tensors
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