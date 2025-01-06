import torch
from typing import List
from src.text_encoder import CTCTextEncoder
from src.metrics.wer import WERMetric,DecodingConfig, DecodingMethod

import torch
from typing import List

def test_wer_decoder_comparison():
    # Создаем инстансы
    text_encoder = CTCTextEncoder()
    
    wer_greedy = WERMetric(
        text_encoder=text_encoder,
        decoding_cfg=DecodingConfig(method=DecodingMethod.GREEDY)
    )
    
    wer_beam = WERMetric(
        text_encoder=text_encoder,
        decoding_cfg=DecodingConfig(method=DecodingMethod.BEAM, beam_size=5)
    )
    
    # Создаем тестовый пример с большей неопределенностью
    sequence_length = 15
    vocab_size = len(text_encoder)
    
    # Создаем два примера в батче
    log_probs = torch.full((2, sequence_length, vocab_size), -10.0)
    
    # Первый пример: "hello world" с неоднозначностями
    # Делаем некоторые буквы неоднозначными, например 'e'/'i', 'o'/'u'
    example1_probs = [
        (text_encoder.char2ind['h'], -0.3),
        (text_encoder.char2ind['e'], -0.4),  # 'e' менее уверенны
        (text_encoder.char2ind['i'], -0.5),  # альтернатива для 'e'
        (text_encoder.char2ind['l'], -0.9),
        (text_encoder.char2ind['l'], -0.3),
        (text_encoder.char2ind['o'], -0.4),  # 'o' менее уверенны
        (text_encoder.char2ind['u'], -0.5),  # альтернатива для 'o'
        (text_encoder.char2ind[' '], -0.3),
        (text_encoder.char2ind['w'], -0.9),
        (text_encoder.char2ind['o'], -0.4),
        (text_encoder.char2ind['r'], -0.3),
        (text_encoder.char2ind['l'], -0.8),
        (text_encoder.char2ind['d'], -0.3),
    ]
    
    # Второй пример: "test case" с другими неоднозначностями
    example2_probs = [
        (text_encoder.char2ind['t'], -0.3),
        (text_encoder.char2ind['e'], -0.7),
        (text_encoder.char2ind['s'], -0.4),
        (text_encoder.char2ind['t'], -0.3),
        (text_encoder.char2ind[' '], -0.3),
        (text_encoder.char2ind['c'], -0.4),
        (text_encoder.char2ind['k'], -0.9),  # альтернатива для 'c'
        (text_encoder.char2ind['a'], -0.3),
        (text_encoder.char2ind['s'], -0.8),
        (text_encoder.char2ind['e'], -0.3),
    ]
    
    # Заполняем вероятности
    for pos, (idx, prob) in enumerate(example1_probs):
        log_probs[0, pos, idx] = prob
    
    for pos, (idx, prob) in enumerate(example2_probs):
        log_probs[1, pos, idx] = prob
    
    # Добавляем blank токены с разными вероятностями
    log_probs[:, :, 0] = -0.6  # blank токены менее вероятны
    
    # Тестируем оба декодера
    texts = ["hello world", "test case"]
    log_probs_length = torch.tensor([13, 10])  # длины последовательностей
    
    print("Input texts:", texts)
    
    # Проверяем декодирование для каждого примера
    for batch_idx in range(2):
        print(f"\nExample {batch_idx + 1}:")
        
        # Для greedy
        pred = log_probs[batch_idx, :log_probs_length[batch_idx]]
        greedy_indices = torch.argmax(pred, dim=-1).numpy()
        greedy_text = text_encoder.ctc_decode(greedy_indices)
        print(f"Greedy decoded text: '{greedy_text}'")
        
        # Для beam search
        beam_text = text_encoder.ctc_decode(pred, beam_size=5)
        print(f"Beam search decoded text: '{beam_text}'")
    
    # Считаем WER для всего батча
    greedy_wer = wer_greedy(log_probs, log_probs_length, texts)
    beam_wer = wer_beam(log_probs, log_probs_length, texts)
    
    print(f"\nOverall Greedy WER: {greedy_wer}")
    print(f"Overall Beam Search WER: {beam_wer}")

# Запускаем тест
test_wer_decoder_comparison()