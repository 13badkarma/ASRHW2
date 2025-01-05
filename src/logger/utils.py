import io

import matplotlib.pyplot as plt
import PIL
from torchvision.transforms import ToTensor

plt.switch_backend("agg")  # fix RuntimeError: main thread is not in main loop


def plot_images(imgs, config):
    """
    Combine several images into one figure.

    Args:
        imgs (Tensor): array of images (B X C x H x W).
        config (DictConfig): hydra experiment config.
    Returns:
        image (Tensor): a single figure with imgs plotted side-to-side.
    """
    # name of each img in the array
    names = config.writer.names
    # figure size
    figsize = config.writer.figsize
    fig, axes = plt.subplots(1, len(names), figsize=figsize)
    for i in range(len(names)):
        # channels must be in the last dim
        img = imgs[i].permute(1, 2, 0)
        axes[i].imshow(img)
        axes[i].set_title(names[i])
        axes[i].axis("off")  # we do not need axis
    # To create a tensor from matplotlib,
    # we need a buffer to save the figure
    buf = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    # convert buffer to Tensor
    image = ToTensor()(PIL.Image.open(buf))

    plt.close()

    return image


# def plot_spectrogram(spectrogram, name=None):
#     """
#     Plot spectrogram

#     Args:
#         spectrogram (Tensor): spectrogram tensor.
#         name (None | str): optional name.
#     Returns:
#         image (Image): image of the spectrogram
#     """
#     plt.figure(figsize=(20, 5))
#     plt.pcolormesh(spectrogram)
#     plt.title(name)
#     buf = io.BytesIO()
#     plt.savefig(buf, format="png")
#     buf.seek(0)

#     # convert buffer to Tensor
#     image = ToTensor()(PIL.Image.open(buf))

#     plt.close()

#     return image

def plot_spectrogram(spectrogram, name=None):
    """
    Plot spectrogram

    Args:
        spectrogram (Tensor): spectrogram tensor [channel, freq, time] or [batch, channel, freq, time]
        name (None | str): optional name.
    Returns:
        image (Image): image of the spectrogram
    """
    # Приводим к нужной размерности
    if spectrogram.dim() == 4:  # если батч
        spectrogram = spectrogram[0]  # берем первый элемент из батча
    if spectrogram.dim() == 3:  # если есть channel dimension
        spectrogram = spectrogram[0]  # берем первый канал
    
    # Переводим в numpy для визуализации
    spectrogram = spectrogram.cpu().numpy()
    
    plt.figure(figsize=(20, 5))
    plt.pcolormesh(spectrogram.T)  # транспонируем для правильного отображения
    if name:
        plt.title(name)
    plt.colorbar(format='%+2.0f dB')
    plt.ylabel('Mel Frequency Bins')
    plt.xlabel('Time')
    
    # Сохраняем в буфер
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    # Конвертируем буфер в тензор
    image = ToTensor()(PIL.Image.open(buf))

    plt.close()

    return image