
from __future__ import print_function
import numpy as np
import os
import sys
import tarfile
import pickle

from urllib.request import urlretrieve
from sklearn.linear_model import logistic

from scipy import ndimage

url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None


def download_proogress_hook(count, block_size, total_size):
    """reports about download progress"""
    global last_percent_reported
    percent = int(count * block_size * 100 / total_size)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

    last_percent_reported = percent


def download(filename, expected_bytes, force=False):
    if force or not os.path.exists(filename):
        print('Attemtping to download:', filename)
        filename, _ = urlretrieve(url + filename, filename, reporthook=download_proogress_hook)
        print('\nDownload Completed!')

    stat_info = os.stat(filename)
    if stat_info.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        raise Exception('Failed to verify '+filename+'. Can you get to it with a browser?')
    return filename


def extract_data(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]
    if os.path.isdir(root) and not force:
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall()
        tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]

    if len(data_folders) != num_classes:
        raise Exception('Expected %d folders, one per class. Found %d instead.'
                        % (num_classes, len(data_folders)))

    print(data_folders)
    return data_folders

# pixel width & height
image_size = 28

# Number of Levels per pixel
pixel_depth = 255.0


def disp_number_images(data_folders):
    for folder in data_folders:
        pickle_fname = ''.join(folder)+'.pickle'
        try:
            with open(pickle_fname, 'rb') as f:
                dataset = pickle.load(f)
        except Exception as e:
            print('Unable to read data from', pickle_fname,':',e)
            return
        print('Number of images in ',folder,' :',len(dataset))


def load_letter(folder, min_num_images):
    """Load data for a single letter"""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size), dtype=np.float32)
    print(dataset.shape)
    print(folder)
    num_images = 0
    for image in image_files:

        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) - pixel_depth/2)/pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images += 1

        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok,skipping.')

    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' % (num_images,min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def my_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names


# Create a validation dataset for hyperparam tuning
def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None

    return dataset, labels


def merge_datasets(pickle_files, train_sz, valid_sz=0):
    num_cls = len(pickle_files)
    valid_ds, valid_lb = make_arrays(valid_sz, image_size)
    train_ds, train_lb = make_arrays(train_sz, image_size)
    vsize_per_class = valid_sz // num_cls
    tsize_per_class = train_sz // num_cls

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class+tsize_per_class
    for label, pickle_fl in enumerate(pickle_files):
        try:
            with open(pickle_fl, 'rb') as fl:

                letter_set = pickle.load(fl)

                # shuffle letters to have random validation and train set
                np.random.shuffle(letter_set)
                if valid_ds is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_ds[start_v:end_v, :, :] = valid_letter
                    valid_lb[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_ds[start_t:end_t, :, :] = train_letter
                train_lb[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class

        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_ds, valid_lb, train_ds, train_lb


# for shuffling test and training distributions
def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

# function calls
train_filename = download('notMNIST_large.tar.gz', 247336696)
test_filename = download('notMNIST_small.tar.gz', 8458043)

num_classes = 10
np.random.seed(133)

train_folders = extract_data(train_filename)
test_folders = extract_data(test_filename)

disp_number_images(train_folders)
disp_number_images(test_folders)

train_datasets = my_pickle(train_folders, 45000)
test_datasets = my_pickle(test_folders, 1800)

train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

# three sets for measuring performance.
print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

pickle_file = 'notMNIST.pickle'

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

cls = logistic.LogisticRegression()

# if you train the full set(20000) put 10000 to valid set size
sample_size = 1000

X_test = test_dataset.reshape(test_dataset.shape[0], 28*28)
y_test = test_labels

X_valid =valid_dataset[:sample_size].reshape(sample_size, 28*28)
y_valid = valid_labels[:sample_size]

X_train = train_dataset[:sample_size].reshape(sample_size, 28*28)
y_train = train_labels[:sample_size]
cls.fit(X_train, y_train)

print(cls.score(X_test,y_test))
pred_labels_test = cls.predict(X_test)
print(pred_labels_test)

print(cls.score(X_valid,y_valid))
pred_labels_valid = cls.predict(X_valid)
print(pred_labels_valid)

