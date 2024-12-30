from torch import nn

class DeepSpeech2(nn.Module):
    """
    DeepSpeech2 model for ASR tasks.
    """

    def __init__(self, n_feats, n_tokens, rnn_hidden=512, num_rnn_layers=5):
        """
        Args:
            n_feats (int): Number of input features (e.g., spectrogram frequencies).
            n_tokens (int): Number of tokens in the vocabulary.
            rnn_hidden (int): Number of hidden units in each RNN layer.
            num_rnn_layers (int): Number of RNN layers.
        """
        super().__init__()

        # Convolutional feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # Conv layer
            nn.BatchNorm2d(32),  # Batch normalization
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # Downsampling
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # Calculate the new feature size after CNN
        cnn_out_size = n_feats // 2  # Adjust for stride in the second conv layer

        # Recurrent layers
        self.rnn = nn.LSTM(
            input_size=cnn_out_size * 32,  # Combine frequency and channel dimensions
            hidden_size=rnn_hidden,
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=True
        )

        # Output projection
        self.fc = nn.Linear(rnn_hidden * 2, n_tokens)  # *2 for bidirectional

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Forward pass through the model.

        Args:
            spectrogram (Tensor): Input spectrogram (batch_size, freq, time).
            spectrogram_length (Tensor): Original lengths of the spectrogram.
        Returns:
            dict: Contains log probabilities and transformed lengths.
        """
        # Add channel dimension (batch_size, 1, freq, time)
        spectrogram = spectrogram.unsqueeze(1)

        # Apply CNN
        cnn_output = self.cnn(spectrogram)

        # Reshape for RNN (batch_size, time, features)
        batch_size, channels, freq, time = cnn_output.shape
        cnn_output = cnn_output.permute(0, 3, 1, 2).reshape(batch_size, time, -1)

        # Apply RNN
        rnn_output, _ = self.rnn(cnn_output)

        # Apply final projection
        output = self.fc(rnn_output)

        # Compute log probabilities
        log_probs = nn.functional.log_softmax(output, dim=-1)

        # Transform input lengths
        log_probs_length = self.transform_input_lengths(spectrogram_length)

        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        Transform input lengths for compressed time dimension.
        Args:
            input_lengths (Tensor): Original input lengths.
        Returns:
            Tensor: Transformed lengths.
        """
        return (input_lengths + 1) // 2  # Adjust for stride in CNN layer

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
