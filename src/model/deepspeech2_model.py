from torch import nn
import torch
import torch.nn.functional as F


class CNNLayerNorm(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        x = x.transpose(2, 3).contiguous()
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()


class ResidualCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super().__init__()
        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x


class BidirectionalGRU(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super().__init__()
        self.BiGRU = nn.GRU(
            input_size=rnn_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=batch_first,
            bidirectional=True
        )
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class DeepSpeech2(nn.Module):
    def __init__(self, n_feats, n_tokens, rnn_hidden=512, num_rnn_layers=5):
        """
    Args:
        n_feats (int): Number of input features (e.g., spectrogram frequencies).
        n_tokens (int): Number of tokens in the vocabulary.
        rnn_hidden (int): Number of hidden units in each RNN layer.
        num_rnn_layers (int): Number of RNN layers.
    """
        super().__init__()

        # Constants
        dropout = 0.1
        n_feats = n_feats // 2
        n_cnn_layers = 3

        # Initial CNN layer
        self.cnn = nn.Conv2d(1, 32, 3, stride=2, padding=3 // 2)

        # Residual CNN layers
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)
            for _ in range(n_cnn_layers)
        ])

        # Fully connected layer
        self.fully_connected = nn.Linear(n_feats * 32, rnn_hidden)

        # Bidirectional GRU layers
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(
                rnn_dim=rnn_hidden if i == 0 else rnn_hidden * 2,
                hidden_size=rnn_hidden,
                dropout=dropout,
                batch_first=i == 0
            )
            for i in range(num_rnn_layers)
        ])

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(rnn_hidden * 2, rnn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden, n_tokens)
        )

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
      Forward pass through the model.

      Args:
          spectrogram (Tensor): Input spectrogram (batch_size, freq, time).
          spectrogram_length (Tensor): Original lengths of the spectrogram.
      Returns:
          dict: Contains log probabilities and transformed lengths.
      """
        # Add channel dimension

        x = spectrogram
        x = self.cnn(x)
        if torch.isnan(x).any():
            print("NaN detected after CNN!")
        x = self.rescnn_layers(x)

        # Reshape for RNN
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        x = x.transpose(1, 2)

        # Apply fully connected and RNN layers
        x = self.fully_connected(x)
        x = self.birnn_layers(x)

        # Final classification
        output = self.fc(x)

        # Compute log probabilities
        log_probs = F.log_softmax(output, dim=-1)
 
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
        return (input_lengths + 3) // 4

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
