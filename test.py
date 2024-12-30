
from src.text_encoder import CTCTextEncoder
from src.datasets.collate import collate_fn
encoder = CTCTextEncoder()
ctc_decoded = encoder.ctc_decode([8, 8, 5, 0, 12, 12, 15])
assert ctc_decoded == "helo"

normalized = encoder.normalize_text("Hello, World!")
assert normalized == "hello world"

ctc_decoded_beam = encoder.ctc_decode([8, 8, 5, 0, 12, 12, 15], beam_size=2)
assert ctc_decoded_beam == "helo"  # Без язык-модели результат должен совпадать с greedy


import torch

def test_collate_fn():
    # Подготовка тестовых данных
    dataset_items = [
        {
            "audio": torch.ones(1, 100),  # Одноканальный аудиосигнал длиной 100
            "spectrogram": torch.ones(128, 1, 50),  # 128 частотных каналов, длина 50
            "text_encoded": torch.tensor([[1, 2, 3]]),  # Закодированный текст
            "text": "abc",
            "audio_path": "/path/to/audio1.wav",
        },
        {
            "audio": torch.ones(1, 80),  # Короткий аудиосигнал
            "spectrogram": torch.ones(128, 1, 40),  # Короткая спектрограмма
            "text_encoded": torch.tensor([[4, 5]]),  # Короткий текст
            "text": "de",
            "audio_path": "/path/to/audio2.wav",
        },
    ]

    # Ожидаемые размеры после батчирования
    result = collate_fn(dataset_items)

    assert result["audio"].shape == (2, 1, 100)  # Batch x Channels x Max_length
    assert result["spectrogram"].shape == (2, 128, 50)  # Batch x Frequency x Max_length
    assert result["text_encoded"].shape == (2, 3)  # Batch x Max_text_length
    assert torch.equal(result["audio_length"], torch.tensor([100, 80]))
    assert torch.equal(result["spectrogram_length"], torch.tensor([50, 40]))
    assert torch.equal(result["text_encoded_length"], torch.tensor([3, 2]))
    assert result["text"] == ["abc", "de"]
    assert result["audio_path"] == ["/path/to/audio1.wav", "/path/to/audio2.wav"]

test_collate_fn()