import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import torch

warnings.filterwarnings("ignore")

from torch.utils import data

from matplotlib import rcParams

from cugdth.geotiff_utils import ReadGeoTiff, CreateGeoTiff
from modelTrain import args
from model import SimCLR
from utils import *

# Define date range for visualization
DateList = pd.date_range('{}-{}'.format(2019, 2), '{}-{}'.format(2024, 6), freq='MS').strftime("%Y-%m")

# Configure matplotlib font settings
rcParams['font.family'] = 'Microsoft YaHei'
rcParams['font.size'] = 13

# Define a constant for NoData values
NO_DATA_VALUE = 32767


def normalize_and_resample(dataset, target_size):
    """
    Normalize and resample the dataset to a target size.
    """
    dataset[dataset == NO_DATA_VALUE] = 0
    dataset = dataset.reshape(-1, args.channels, dataset.shape[1], dataset.shape[2])
    dataset = dataset[:, :args.channels]  # Keep only the first 10 bands
    dataset = standardize_data(dataset)  # Standard
    dataset = dataset.reshape(-1, dataset.shape[2], dataset.shape[3])
    dataset = resize_image(dataset, target_size)
    return dataset


class MaskDataset(data.Dataset):
    """
    Custom dataset class for time series data.
    """

    def __init__(self, dataset, indices, sequence_length):
        super(MaskDataset, self).__init__()
        self.indices = indices
        self.dataset = dataset
        self.sequence_length = sequence_length

    def __getitem__(self, index):
        start_idx = self.indices[index]
        data_slice = self.dataset[start_idx:start_idx + self.sequence_length]
        return data_slice.reshape(-1, data_slice.shape[2], data_slice.shape[3])

    def __len__(self):
        return len(self.indices)


def generate_rgb_image(data, mask):
    """
    Convert multi-channel data to an RGB image.
    """
    data = data.astype('int32')
    red, green, blue = data[2], data[1], data[0]

    # Apply percentile-based normalization for better visualization
    min_r, max_r = np.percentile(red[mask == 1], [2, 98])
    min_g, max_g = np.percentile(green[mask == 1], [2, 98])
    min_b, max_b = np.percentile(blue[mask == 1], [2, 98])

    red = (red - min_r) * 255 / (max_r - min_r)
    green = (green - min_g) * 255 / (max_g - min_g)
    blue = (blue - min_b) * 255 / (max_b - min_b)

    result = np.stack([red, green, blue])
    result[:, mask == 0] = 0  # Set border pixels to black
    return result.astype('int32')


def extract_images_from_path(path, indices):
    """
    Extract images from dataset for visualization.
    """
    _, source_data = process_data(path)
    images = []
    for idx, data in enumerate(source_data):
        if idx in indices:
            mask = np.max(source_data[idx, :10], axis=0) != 0
            images.append(generate_rgb_image(source_data[idx], mask).transpose(1, 2, 0))
    return images


def perform_model_inference(data_path, model_path):
    """
    Perform inference using the trained model.
    """
    input_dim = args.channels * args.sample_size  # Compute input dimension
    model = SimCLR.SimCLR(input_dim, args.proj_size).to(args.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    data_loader, data_split_count = get_data_loader(data_path, args)

    results = torch.empty([0, args.proj_size])
    for batch_data in data_loader:
        batch_data = batch_data.to(args.device).float()
        _, features = model(batch_data)
        results = torch.cat((results, features.cpu().detach()), dim=0)

        if results.shape[0] >= data_split_count:
            final_results = results[:data_split_count]
            similarity_curves, transition_intervals = detect_transition_intervals(final_results, args.sample_size,
                                                                                     args.threshold, 0)
            return similarity_curves, transition_intervals


def process_data(path):
    """
    Read and preprocess geospatial data.
    """
    im_data, _, _ = ReadGeoTiff(path)
    resampled_data = normalize_and_resample(im_data, args.block_size)
    resampled_data = resampled_data.reshape(args.len_ts, args.channels, resampled_data.shape[1],
                                            resampled_data.shape[2])
    source_data = im_data.reshape(args.len_ts, args.channels, im_data.shape[1], im_data.shape[2])
    return resampled_data, source_data


def get_data_loader(path, args):
    """
    Prepare dataset and dataloader for inference.
    """
    resampled_data, _ = process_data(path)
    indices = list(range(0, args.len_ts - args.sample_size + 1))
    dataset = MaskDataset(resampled_data, indices, args.sample_size)
    data_loader = data.DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)
    return data_loader, len(indices)


def visualize_results(data_path, model_path, selected_indices):
    """
    Generate and display model inference results along with image visualization.
    """
    plt.figure(figsize=(12, 3))
    gs = gridspec.GridSpec(2, 9)

    plt.subplot(gs[:2, :6])
    plt.grid(alpha=0.3)
    similarity_curves, transition_intervals = perform_model_inference(data_path, model_path)
    plt.plot(range(5, 59), similarity_curves, marker='o', ms=3, color='black', label='SSC-SB')

    x = np.linspace(5, 58, 54)
    for idx, interval in enumerate(transition_intervals):
        start, end = interval[0] + 5, interval[-1] + 6
        plt.fill_between(x, -1, 1, where=(x >= start) & (x <= end), color='red', alpha=0.4,
                         label='Transition zone' if idx == 0 else "")
        plt.axvline(start, color='red', ls='--', alpha=0.8)
        plt.axvline(end, color='red', ls='--', alpha=0.8)
    for i, show in enumerate(show_list):
        try:
            y = similarity_curves[show - 5]
        except:
            y = 1
        if i == 0:
            plt.scatter(show, y, color='red', zorder=4, s=40, label='Image visualization')
        else:
            plt.scatter(show, y, color='red', zorder=4, s=40)
    date_list = np.array([4, 16, 28, 40, 52, 64])
    plt.xticks(date_list, DateList[date_list])
    plt.ylim(-1.1, 1.1)
    plt.legend()

    images = extract_images_from_path(data_path, selected_indices)
    for img, row, col, date_idx in zip(images, [0, 0, 0, 1, 1, 1], [6, 7, 8, 6, 7, 8], selected_indices):
        plt.subplot(gs[row, col])
        plt.imshow(img)
        plt.xlabel(DateList[date_idx])
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    model_path = os.path.join(args.model_save_folder, 'Latest_model.pth')

    demo_path = rf'dataset\street_demo\development.tif'
    show_list = [35, 39, 40, 43, 46, 48]
    visualize_results(demo_path, model_path, show_list)

    demo_path = rf'dataset\street_demo\stable.tif'
    show_list = [5, 15, 25, 35, 45, 55]
    visualize_results(demo_path, model_path, show_list)
