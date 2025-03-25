import os
import numpy as np
from scipy.ndimage import zoom

def create_directory(path):
    """Create a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)



def resize_image(data, target_size):
    """Resize the image data to the specified target size."""
    scale_height = target_size / data.shape[1]
    scale_width = target_size / data.shape[2]

    # Resample the data with the computed scale factors
    resized_data = zoom(data, (1, scale_height, scale_width), order=1)

    # Ensure the correct shape after resizing
    resized_data = resized_data.reshape(-1, int(target_size), int(target_size))
    return resized_data


def standardize_data(data):
    """Standardize the dataset using predefined mean and standard deviation values."""
    data = data.astype('float')
    mean_values = np.array([852.97, 1053.48, 1061.41, 1342.57, 1834.50,
                            2025.81, 2072.27, 2128.08, 1875.54, 1477.33])
    std_values = np.array([431.51, 446.31, 551.38, 513.61, 648.89,
                           736.67, 808.45, 793.00, 750.65, 753.42])

    # Normalize each channel separately
    for idx, img in enumerate(data):
        for channel_idx, channel_data in enumerate(img):
            channel_data = (channel_data - mean_values[channel_idx]) / std_values[channel_idx]
            img[channel_idx] = channel_data
        data[idx] = img

    # Apply mask to remove invalid areas (where original data is 0)
    mask = np.mean(data[0], axis=0) == 0
    data[:, :, mask] = 0
    return data


def pearson_correlation(vec1, vec2):
    """Compute the Pearson correlation coefficient between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    vec1_centered = vec1 - np.mean(vec1)
    vec2_centered = vec2 - np.mean(vec2)
    return np.dot(vec1_centered, vec2_centered) / (np.linalg.norm(vec1_centered) * np.linalg.norm(vec2_centered))


def detect_transition_intervals(feature_series, window_size, threshold=0.9, offset=0):
    """Identify transition intervals based on similarity scores within a time series of street features."""
    similarity_scores = []  # Stores Pearson similarity values over time
    transition_intervals = []  # Stores detected transition periods
    current_interval = []  # Temporary list for tracking ongoing transitions

    for idx, feature_vector in enumerate(feature_series):
        try:
            # Compute similarity between current and future time steps
            similarity = float(pearson_correlation(feature_series[idx], feature_series[idx + window_size - offset]))
            future_similarity = float(
                pearson_correlation(feature_series[idx + 1], feature_series[idx + window_size + 1 - offset]))

            similarity_scores.append(similarity)

            # Detect transition zones based on similarity threshold
            if (similarity < threshold) or (
                    idx - 1 in current_interval and idx - 2 in current_interval and future_similarity < threshold):
                current_interval.append(idx)
            elif similarity > threshold:
                if len(current_interval) != 0:
                    transition_intervals.append(current_interval)
                    current_interval = []
        except IndexError:
            if len(current_interval) != 0:
                transition_intervals.append(current_interval)
            break

    # Filter transitions based on length criteria
    filtered_intervals = []
    for interval in transition_intervals:
        if len(interval) >= 4 or (len(similarity_scores) - 1 in interval and len(interval) >= 2):
            filtered_intervals.append(interval)

    return similarity_scores, filtered_intervals
