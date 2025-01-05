import torch
import torchaudio
from src.model.deepspeech2_model import DeepSpeech2
import torch.nn.functional as F

def collate_fn(batch):
    """
    Создает батч из списка аудио тензоров разной длины
    """
    # Находим максимальную длину в батче
    max_length = max(waveform.shape[-1] for waveform in batch)
    
    # Паддинг каждого семпла до максимальной длины
    padded_batch = []
    lengths = []
    
    for waveform in batch:
        pad_length = max_length - waveform.shape[-1]
        # Паддим справа нулями
        padded = F.pad(waveform, (0, pad_length))
        padded_batch.append(padded)
        lengths.append(waveform.shape[-1])
    
    # Стакаем все в один тензор
    padded_batch = torch.stack(padded_batch)
    lengths = torch.tensor(lengths)
    
    return padded_batch, lengths

def test_batch_audio():
    # Параметры
    n_mels = 128
    sample_rate = 16000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
    
    # Загрузка аудио файлов
    waveforms = []
    for path in audio_paths:
        waveform, sr = torchaudio.load(path)
        assert sr == sample_rate, f"Sample rate mismatch in {path}: {sr} != {sample_rate}"
        waveforms.append(waveform)
    
    # Создаем батч с паддингом
    padded_waveforms, waveform_lengths = collate_fn(waveforms)
    
    # Создание процессора признаков
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate
    ).to(device)
    
    # Перемещаем данные на устройство
    padded_waveforms = padded_waveforms.to(device)
    
    # Получение спектрограмм (обрабатываем весь батч сразу)
    spectrograms = mel_spectrogram(padded_waveforms).squeeze(1)
    
    # Длины спектрограмм (после MelSpectrogram длины изменяются)
    spec_lengths = torch.ceil(waveform_lengths.float() / mel_spectrogram.hop_length).int()
    
    # Создание модели
    model = DeepSpeech2(
        n_feats=n_mels,
        n_tokens=29,
        rnn_hidden=512,
        num_rnn_layers=5
    ).to(device)
    
    # Переключение в режим оценки

    model.eval()
    
    # Прямой проход
    with torch.no_grad():
        output = model(
            spectrogram=spectrograms,
            spectrogram_length=spec_lengths
        )
    
    # Вывод результатов
    print(f"Batch size: {len(audio_paths)}")
    print(f"Input spectrograms shape: {spectrograms.shape}")
    print(f"Output log probs shape: {output['log_probs'].shape}")
    print(f"Output lengths: {output['log_probs_length']}")
    
    return output

if __name__ == "__main__":
    test_batch_audio()