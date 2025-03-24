import sys
import logging
import os 
import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import cv2

CIFAR10_TRAIN_MEAN = np.array((0.4914, 0.4822, 0.4465))[None, :, None, None]
CIFAR10_TRAIN_STD = np.array((0.2470, 0.2435, 0.2616))[None, :, None, None]


def get_logger(filename):
    # Logging configuration: set the basic configuration of the logging system
    log_formatter = logging.Formatter(fmt='%(asctime)s [%(levelname)-5.5s] %(message)s',
                                      datefmt='%Y-%b-%d %H:%M')
    logger = logging.getLogger()
    # **Avoid adding multiple handlers** by checking if they exist
    if logger.hasHandlers():
        logger.handlers.clear()  # **Clear existing handlers**

    logger.setLevel(logging.INFO)  #DEBUG)
    # File logger
    file_handler = logging.FileHandler(filename, mode='w')  # default is 'a' to append
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)  #DEBUG)
    logger.addHandler(file_handler)
    # Stdout logger
    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(log_formatter)
    std_handler.setLevel(logging.INFO) #DEBUG)
    logger.addHandler(std_handler)
    return logger


############################################ for folder type images #####################################################################

def resize_image(image_path, config):     
    """Resize an image to the specified target size."""
    target_size = config.image_size
    image = Image.open(image_path)
    resized_image = image.resize(target_size, Image.LANCZOS)
    return np.array(resized_image)

def get_data_folder(config):
    if config.dataset == 'medical_images_2' or config.dataset == 'face_images_2':
        
        data_folder = os.path.join(config.data_path, config.dataset)
        
        train_X, train_y = [], []
        test_X, test_y = [], []

        class_names = os.listdir(data_folder)
        class_names = [name for name in class_names if os.path.isdir(os.path.join(data_folder, name))]
        
        encoder = LabelEncoder()
        encoder.fit(class_names)

        # Print the mapping of class names to integers
        class_to_int_mapping = dict(zip(class_names, encoder.transform(class_names)))
        print("Class to integer mapping:")
        for class_name, label in class_to_int_mapping.items():
            print(f"Class Name: {class_name}, Integer Label: {label}")    

        for class_name in class_names: 
            class_folder = os.path.join(data_folder, class_name)
            if os.path.isdir(class_folder):
                images = []
                for file_name in os.listdir(class_folder):
                    if file_name.endswith(('.tif', '.png', '.jpg')):
                        image_path = os.path.join(class_folder, file_name)
                        resized_image = resize_image(image_path, config)
                        images.append(resized_image)
                np.random.shuffle(images)
                num_train = int(len(images) * config.train_split)
                train_X.extend(images[:num_train])
                train_y.extend([encoder.transform([class_name])[0]] * num_train)  #train_y.extend([int(class_name)] * num_train)
                test_X.extend(images[num_train:])
                test_y.extend([encoder.transform([class_name])[0]] * (len(images) - num_train))  #test_y.extend([int(class_name)] * (len(images) - num_train))

        train_X = np.array(train_X)
        train_y = np.array(train_y, dtype=np.int64)
        test_X = np.array(test_X)
        test_y = np.array(test_y, dtype=np.int64)

        # Print shapes before transpose
        #print(f"train_X shape before transpose: {train_X.shape}")
        #print(f"test_X shape before transpose: {test_X.shape}")

        # Transpose the arrays to convert them from NHWC to NCHW format
        train_X = np.transpose(train_X, (0, 3, 1, 2))
        test_X = np.transpose(test_X, (0, 3, 1, 2))

        # Print the shapes of the arrays
        #print(f"train_X shape: {train_X.shape}")
        #print(f"train_y shape: {train_y.shape}")
        #print(f"test_X shape: {test_X.shape}")
        #print(f"test_y shape: {test_y.shape}")

    else:
        raise ValueError("Unknown dataset")

    return train_X, train_y, test_X, test_y

############################################ for face images in npz format ###############################################################
def get_data_npz(config):

    if config.dataset == 'mnist' or config.dataset == 'fashion-mnist' or config.dataset == 'org_images':

        data_file = f"{config.data_path}/{config.dataset}.npz"
       # print(f"Loading data from: {data_file}")
        dataset = np.load(data_file)
        train_X, train_y = dataset['x_train'], dataset['y_train'].astype(np.int64)
        test_X, test_y = dataset['x_test'], dataset['y_test'].astype(np.int64)

        if config.dataset == 'fashion-mnist':
            train_X = np.reshape(train_X, (-1, 1, 28, 28))
            test_X = np.reshape(test_X, (-1, 1, 28, 28))
        else:
            train_X = np.transpose(train_X, (0, 3, 1, 2))
            test_X = np.transpose(test_X, (0, 3, 1, 2))

        # Print some information about the loaded data
        #print(f"Train data shape: {train_X.shape}, Train labels shape: {train_y.shape}")
        #print(f"Test data shape: {test_X.shape}, Test labels shape: {test_y.shape}")        
    else:

        raise ValueError("Unknown dataset")

    return train_X, train_y, test_X, test_y

def data_loader(dataset, inputs, targets, batch_size, is_train=True):
# if you want try different normalization merthods 
    def cifar10_norm(x):
        #x -= CIFAR10_TRAIN_MEAN
        #x /= CIFAR10_TRAIN_STD
        # Calculate mean and standard deviation along the (0, 2, 3) axes
        mean = np.mean(x, axis=(0, 2, 3))
        std = np.std(x, axis=(0, 2, 3))

        # Normalize the data using the computed mean and standard deviation
        x = (x - mean.reshape(1, -1, 1, 1)) / std.reshape(1, -1, 1, 1)
        return x

    def no_norm(x):
        return x
    
    def min_max_norm(x):
        """This scales the pixel intensity values to a fixed range, typically [0, 1].
        Advantages:
        Retains original intensity patterns.
        Works well when the pixel intensity range is consistent across images. """
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    
    def hist_equalization(img):
        """ Enhances the contrast of images by redistributing the pixel intensities uniformly. This is useful for images with poor contrast.
        Advantages:
        Enhances image contrast, making features more distinguishable.
        Useful for grayscale cellular images."""
        if len(img.shape) == 2:  # Grayscale image
            return cv2.equalizeHist(img)
        elif len(img.shape) == 3:  # Color image
            ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        
    def clahe_equalization(img):
        """ A more sophisticated version of histogram equalization that limits contrast amplification to avoid noise.
        Advantages:
        Better control of noise amplification.
        Ideal for medical images with subtle features. """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(img.shape) == 2:  # Grayscale
            return clahe.apply(img)
        elif len(img.shape) == 3:  # Color
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    def percentile_norm(x):
        """ This approach removes outlier effects by clipping pixel intensities to the 1st and 99th percentiles, followed by min-max scaling.
        Advantages:
        Reduces the impact of extreme pixel values.
        Effective for datasets with outliers. """
        lower = np.percentile(x, 1)
        upper = np.percentile(x, 99)
        x = np.clip(x, lower, upper)
        return (x - lower) / (upper - lower)

    def log_transform(x):
        """ This enhances low-intensity values by compressing the dynamic range of higher intensities. Common for CT or fluorescence microscopy images.
        Advantages:
        Highlights small intensity variations.
        Suitable for data with a large dynamic range. For fluorescence microscopy or high-dynamic-range images."""
        return np.log1p(x)  # Adds 1 to avoid log(0)

    if dataset == 'org_images' or dataset == 'medical_images_2' or dataset == 'face_images_2':  #'cifar10':
        norm_func = cifar10_norm
    else:
        norm_func = no_norm

    assert inputs.shape[0] == targets.shape[0]
    n_examples = inputs.shape[0]

    sample_rate = batch_size / n_examples
    num_blocks = int(n_examples / batch_size)
    if is_train:
        for i in range(num_blocks):
            mask = np.random.rand(n_examples) < sample_rate
            if np.sum(mask) != 0:
                normalized_inputs = norm_func(inputs[mask].astype(np.float32) / 255.)
                #print(f"Train Batch {i + 1}: Input shape {normalized_inputs.shape}, Target shape {targets[mask].shape}")
                yield (normalized_inputs, targets[mask])
    else:
        for i in range(num_blocks):
            data_batch = norm_func(inputs[i * batch_size: (i+1) * batch_size].astype(np.float32) / 255.)
            target_batch = targets[i * batch_size: (i+1) * batch_size]
            #print(f"Test Batch {i + 1}: Data shape {data_batch.shape}, Target shape {target_batch.shape}")
            yield (data_batch, target_batch)
        if num_blocks * batch_size != n_examples:
            yield (norm_func(inputs[num_blocks * batch_size:].astype(np.float32) / 255.),
                   targets[num_blocks * batch_size:])


def partition_data(train_X, train_y, config):
    # Generate an array of indices for shuffling
    idx = np.arange(0, len(train_X))
    np.random.shuffle(idx)

    # Select the first n_client_data samples
    client_data_idx = idx[:config.n_client_data]
    # Extract client data and labels based on selected indices
    client_data = train_X[client_data_idx], train_y[client_data_idx]

    # Debug: Print the shape of the selected client data and labels
    print(f"Selected client data shape: {client_data[0].shape}")
    print(f"Selected client labels shape: {client_data[1].shape}")

    return client_data

def evaluate_model(model, data, config):
# to get the accuracy 
    model.eval()
    x, y = data

    loader = data_loader(config.dataset, x, y, batch_size=1000, is_train=False)
    acc = 0.
    for xt, yt in loader:
        xt = torch.tensor(xt, requires_grad=False, dtype=torch.float32).to(config.device)
        yt = torch.tensor(yt, requires_grad=False, dtype=torch.int64).to(config.device)
        preds_labels = torch.squeeze(torch.max(model(xt), 1)[1])
        acc += torch.sum(preds_labels == yt).item()

    return acc / x.shape[0]


def extract_numpy_weights(model):
# extract weights from the model in numpy format 
    tensor_weights = model.state_dict()
    numpy_weights = {}

    for k in tensor_weights.keys():
        numpy_weights[k] = tensor_weights[k].detach().cpu().numpy()

    return numpy_weights
  
def convert_np_weights_to_tensor(weights):
    """
    Convert weights from NumPy arrays or PyTorch tensors to PyTorch tensors.
    """
    for k in weights.keys():
        if isinstance(weights[k], np.ndarray):
            weights[k] = torch.from_numpy(weights[k])
        elif isinstance(weights[k], torch.Tensor):
            pass  # Already a torch.Tensor, no need to convert
        else:
            raise TypeError(f"Unsupported type for weights[{k}]: {type(weights[k])}. Must be np.ndarray or torch.Tensor.")
    return weights   

# Function for Random Label Flipping (Data Poisoning Attack)
def apply_random_label_flipping(labels, num_classes):
    """
    Data Poisoning Attack: Assigns random labels to the dataset.

    Args:
        labels (numpy.ndarray or torch.Tensor): Original labels.
        num_classes (int): Total number of classes.

    Returns:
        numpy.ndarray or torch.Tensor: Poisoned labels with random class assignments.
    """
    poisoned_labels = np.random.randint(0, num_classes, size=labels.shape)
    return poisoned_labels
