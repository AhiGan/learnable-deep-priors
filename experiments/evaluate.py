import sys
sys.path.append('../src')
import numpy as np
import os
from evaluation import load_dataset, load_results, convert_results, compute_ami_score, compute_mse_score, plot_samples
import matplotlib.pyplot as plt

def print_scores(folder, filename_base='result_{}.h5', num_tests=5, valid_all=True):
    ami_list, mse_list = [], []
    for model_id in range(num_tests):
        results = load_results(folder, filename_base.format(model_id))
        gamma, predictions, reconstructions = convert_results(results)
        ami_list.append(compute_ami_score(predictions, labels_ami))
        mse_list.append(compute_mse_score(reconstructions, labels_mse, valid_all))
    print('AMI: {:.3f}\tMSE: {:.2f}e-2'.format(np.mean(ami_list), np.mean(mse_list) * 1e2))

def plot_results(folder, images, filename_base='result_{}.h5', num_tests=5):
    for model_id in range(num_tests):
        results = load_results(folder, filename_base.format(model_id))
        gamma, _, reconstructions = convert_results(results)
        plot_samples(model_id,images, gamma, reconstructions)

folder_data = '../data'


# # Multi-Shapes 20x20
# images, labels_ami, labels_mse = load_dataset(folder_data, 'shapes_20x20')
# state_size, updater_size = 16, 32
# folder = os.path.join('shapes_20x20', '{}_{}'.format(state_size, updater_size))
# print_scores(folder, valid_all=False)
#
# # Multi-MNIST
# for variants in ['20', '500', 'all']:
#     print('{} Variants'.format(variants if variants != 'all' else 70000))
#     images, labels_ami, labels_mse = load_dataset(folder_data, 'mnist_{}'.format(variants))
#     for state_size, updater_size in [(16, 32), (32, 32), (32, 64), (64, 64)]:
#         folder = os.path.join('mnist_{}'.format(variants), '{}_{}'.format(state_size, updater_size))
#         print('F={}, N={}'.format(updater_size, state_size), end='\t')
#         print_scores(folder)
#
#
# # Generalization
# '''
# 2 Objects
# AMI: 0.961	MSE: 0.11e-2
# 3 Objects
# AMI: 0.941	MSE: 0.28e-2
# 4 Objects
# AMI: 0.900	MSE: 0.69e-2
# '''
# for num_objects in [2, 3, 4]:
#     print('{} Objects'.format(num_objects))
#     images, labels_ami, labels_mse = load_dataset(folder_data, 'shapes_28x28_{}'.format(num_objects))
#     state_size, updater_size = 64, 64
#     folder = os.path.join('shapes_28x28_1', '{}_{}'.format(state_size, updater_size))
#     filename_base='general_{}_result_{{}}.h5'.format(num_objects) if num_objects != 3 else 'result_{}.h5'
#     print_scores(folder, filename_base)


# Plot Shapes 20x20
images, labels_ami, labels_mse = load_dataset(folder_data, 'shapes_20x20')
state_size, updater_size = 16, 32
folder = os.path.join('shapes_20x20', '{}_{}'.format(state_size, updater_size))
# imgesname = 'shapes_20x20'
plot_results(folder, images)

#
# # Plot Shapes 28x28
# images, labels_ami, labels_mse = load_dataset(folder_data, 'shapes_28x28_3')
# state_size, updater_size = 64, 64
# folder = os.path.join('shapes_28x28_1', '{}_{}'.format(state_size, updater_size))
# plot_results(folder, images)
#
# # Plot MNIST 20
# images, labels_ami, labels_mse = load_dataset(folder_data, 'mnist_20')
# state_size, updater_size = 64, 64
# folder = os.path.join('mnist_20', '{}_{}'.format(state_size, updater_size))
# plot_results(folder, images)
#
# # Plot MNIST 500
# images, labels_ami, labels_mse = load_dataset(folder_data, 'mnist_500')
# state_size, updater_size = 32, 64
# folder = os.path.join('mnist_500', '{}_{}'.format(state_size, updater_size))
# plot_results(folder, images)
#
# # Plot MNIST 70000
# images, labels_ami, labels_mse = load_dataset(folder_data, 'mnist_all')
# state_size, updater_size = 64, 64
# folder = os.path.join('mnist_all', '{}_{}'.format(state_size, updater_size))
# plot_results(folder, images)