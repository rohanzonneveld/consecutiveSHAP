import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

# Define your paths
path1 = 'experiments/MöbiusHEDGE/SST-2'
path2 = 'experiments/HEDGE/SST-2'
image = 'visualization_sentence'

# Generate list of image pairs
image_pairs = [(f'{path1}/{folder}/{image}_{folder}.png', f'{path2}/{folder}/{image}_{folder}.png') for folder in os.listdir(path1)]
image_pairs = sorted(image_pairs, reverse=False)
# Initialize the index
index = 0

# Function to show images
def show_images(index):
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    
    img1 = mpimg.imread(image_pairs[index][0])
    img2 = mpimg.imread(image_pairs[index][1])
    
    ax1.imshow(img1)
    ax1.set_title(f'MöbiusHEDGE {os.path.basename(image_pairs[index][0])[-9:-4]}')
    ax1.axis('off')
    
    ax2.imshow(img2)
    ax2.set_title(f'HEDGE')
    ax2.axis('off')
    
    plt.show()


for index in range(len(image_pairs)):
    show_images(index)
