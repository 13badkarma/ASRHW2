from torch import nn
import torch

def collate_fn(dataset_items: list[dict]):
    spectrograms = []
    audio_signals = []
    text_encoded_list = []
    
    # Списки для метаданных
    audio_lengths = []
    spectrogram_lengths = []
    text_encoded_lengths = []
    texts = []
    audio_paths = []

    for item in dataset_items:
        # Аудио
        audio = item["audio"]  # [1, time]
        audio_signals.append(audio.squeeze(0))  # [time]
        audio_lengths.append(audio.shape[1])

        # Спектрограмма - меняем порядок обработки
        spec = item["spectrogram"]  # [1, freq=128, time]
        spec = spec.squeeze(0)  # [freq=128, time]
        # Не делаем permute здесь, чтобы сохранить правильный порядок данных
        spectrograms.append(spec)
        spectrogram_lengths.append(spec.shape[1])  # длина по времени

        # Текст
        text_encoded = item["text_encoded"]
        text_encoded_list.append(text_encoded.squeeze(0))
        text_encoded_lengths.append(text_encoded.shape[1])

        texts.append(item["text"])
        audio_paths.append(item["audio_path"])

    # Паддинг последовательностей
    audio_batch = nn.utils.rnn.pad_sequence(audio_signals, batch_first=True)  # [batch, time]
    audio_batch = audio_batch.unsqueeze(1)  # [batch, channel=1, time]

    # Для спектрограмм создаем батч напрямую
    max_spec_len = max(spec.shape[1] for spec in spectrograms)
    spec_batch = torch.zeros(len(spectrograms), 1, 128, max_spec_len, 
                           device=spectrograms[0].device)
    
    for i, spec in enumerate(spectrograms):
        spec_len = spec.shape[1]
        # Явно указываем, что паддинг должен быть только по временной оси
        spec_batch[i, 0, :128, :spec_len] = spec[:128, :spec_len]

    text_encoded_batch = nn.utils.rnn.pad_sequence(text_encoded_list, batch_first=True)

    # Конвертируем длины в тензоры
    audio_lengths = torch.tensor(audio_lengths)
    spectrogram_lengths = torch.tensor(spectrogram_lengths)
    text_encoded_lengths = torch.tensor(text_encoded_lengths)

    return {
        "audio": audio_batch,
        "audio_length": audio_lengths,
        "spectrogram": spec_batch,  # [batch, channel=1, freq=128, time]
        "spectrogram_length": spectrogram_lengths,
        "text_encoded": text_encoded_batch,
        "text_encoded_length": text_encoded_lengths,
        "text": texts,
        "audio_path": audio_paths,
    }