# Based on seminar materials

# Don't forget to support cases when target_text == ''

def _levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein distance between two strings

    Args:
        s1: First string
        s2: Second string
    Returns:
        Minimum number of single-character edits needed to transform s1 into s2
    """
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def calc_cer(target_text: str, predicted_text: str) -> float:
    """
    Calculate Character Error Rate between target and predicted texts

    CER = (insertions + deletions + substitutions) / length of target text

    Args:
        target_text: Ground truth text
        predicted_text: Model prediction
    Returns:
        Character error rate (float between 0 and 1)
    """
    # Handle empty string cases
    if len(target_text) == 0:
        return 1.0 if len(predicted_text) > 0 else 0.0

    # Calculate Levenshtein distance
    distance = _levenshtein_distance(target_text, predicted_text)

    # Normalize by target length
    return float(distance) / len(target_text)

def calc_wer(target_text: str, predicted_text: str) -> float:
    """
    Calculate Word Error Rate between target and predicted texts

    WER = (insertions + deletions + substitutions) / number of words in target

    Args:
        target_text: Ground truth text
        predicted_text: Model prediction
    Returns:
        Word error rate (float between 0 and 1)
    """
    # Handle empty string cases
    if len(target_text) == 0:
        return 1.0 if len(predicted_text) > 0 else 0.0

    # Split into words
    target_words = target_text.split()
    predicted_words = predicted_text.split()

    if len(target_words) == 0:
        return 1.0 if len(predicted_words) > 0 else 0.0

    # Calculate Levenshtein distance on words
    distance = _levenshtein_distance(target_words, predicted_words)

    # Normalize by number of words in target
    return float(distance) / len(target_words)