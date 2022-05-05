import os
from io import StringIO

# if using anaconda3 and error execute: conda install --channel conda-forge pillow=5.2.0
import numpy as np

import huffman
import lzw
import util
from channel import channel
from imageSource import ImageSource
from unireedsolomon import rs
from util import Time

from collections import Counter

import math

# ========================= SOURCE =========================
IMG_NAME = 'image.jpg'

dir_path = os.path.dirname(os.path.realpath(__file__))
IMG_PATH = os.path.join(dir_path, IMG_NAME)  # use absolute path

print(F"Loading {IMG_NAME} at {IMG_PATH}")
img = ImageSource().load_from_file(IMG_PATH)
print(img)
# uncomment if you want to display the loaded image
# img.show()
# uncomment if you want to show the histogram of the colors
# im+g.show_color_hist()

# ================================================================

# ======================= SOURCE ENCODING ========================
# =========================== Huffman ============================
t = Time()

# generating huffman tree
t.tic()
pixels = img.get_pixel_seq()
huffman_freq = Counter(pixels).items()
huffman_tree = huffman.Tree(huffman_freq)
print(F"Generating the Huffman Tree took {t.toc_str()}")

# encoding message
t.tic()
encoded_message_hm = huffman.encode(huffman_tree.codebook, img.get_pixel_seq())
print("length message", len(encoded_message_hm))
print("Enc: {}".format(t.toc()))

t.tic()
received_message = channel(encoded_message_hm, ber=0.01)
t.toc_print()

t.tic()
decoded_message = huffman.decode(huffman_tree, received_message)
print("Dec: {}".format(t.toc()))

decoded_message = decoded_message[:len(pixels)]
decoded_message = np.array(decoded_message)
img.from_bitmap(decoded_message)
img.show()
