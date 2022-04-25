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

# ====================== CHANNEL ENCODING ========================
# ======================== Reed-Solomon ==========================

# as we are working with symbols of 8 bits
# choose n such that m is divisible by 8 when n=2^m−1
# Example: 255 + 1 = 2^m -> m = 8
n = 255  # code_word_length in symbols
k = 223  # message_length in symbols

coder = rs.RSCoder(n, k)

# TODO generate a matrix with k symbols per rows (for each message)
# TODO afterwards you can iterate over each row to encode the message

messages = []

# calculate desired length of the message
length_message = 8*223*math.ceil(len(encoded_message_hm) / (8 * 223))

# calculate amount of padding
amount_padding = length_message - len(encoded_message_hm)

# add padding
encoded_message_hm = encoded_message_hm.zfill(length_message)

encoded_message_hm_uint8 = util.bit_to_uint8(encoded_message_hm)

encoded_message_hm_str = ""
for element in encoded_message_hm_uint8:
    encoded_message_hm_str += chr(element)

encoded_message_hm_blocks = [encoded_message_hm_str[idx: idx + k] for idx in range(0, len(encoded_message_hm_str), 223)]

for message in encoded_message_hm_blocks:
    messages.append(message)

rs_encoded_message = StringIO()
t.tic()
for message in messages:
    code = coder.encode_fast(message, return_string=True)
    rs_encoded_message.write(code)

# TODO What is the RSCoder outputting? Convert to a uint8 (byte) stream before putting it over the channel
# output is een string
rs_encoded_message_uint8 = np.array(
    [ord(c) for c in rs_encoded_message.getvalue()], dtype=np.uint8)
print(t.toc())
print("ENCODING COMPLETE")

# TODO Use this helper function to convert a uint8 stream to a bit stream
rs_encoded_message_bit = util.uint8_to_bit(rs_encoded_message_uint8)

t.tic()
received_message = channel(rs_encoded_message_bit, ber=0.001)
t.toc_print()

# TODO Use this helper function to convert a bit stream to a uint8 stream
received_message_uint8 = util.bit_to_uint8(received_message)

received_message_uint8 = np.split(received_message_uint8, len(messages))

rs_encoded_message_uint8 = np.split(rs_encoded_message_uint8, len(messages))

decoded_message = StringIO()

t.tic()
received_message_str = ""
# TODO Iterate over the received messages and compare with the original RS-encoded messages
for cnt, (block, original_block) in enumerate(zip(received_message_uint8, rs_encoded_message_uint8)):
    try:
        decoded, ecc = coder.decode_fast(block, return_string=True)
        assert coder.check(decoded + ecc), "Check not correct"
        decoded_message.write(str(decoded))
        leng = len(decoded)
        print("count", cnt, len(decoded))
        received_message_str += str(decoded)
    except rs.RSCodecError as error:
        diff_symbols = len(block) - (original_block == block).sum()
        print(
            F"Error occured after {cnt} iterations of {len(received_message_uint8)}")
        print(F"{diff_symbols} different symbols in this block")

t.toc_print()

# received_message_str = [received_message_str[idx: idx + 1] for idx in range(0, len(received_message_str), 1)]
received_message_uint8 = np.array(
    [ord(c) for c in decoded_message.getvalue()], dtype=np.uint8)

received_message_end = util.uint8_to_bit(received_message_uint8)

received_message_end = received_message_end[amount_padding:]

t.tic()
decoded_message = huffman.decode(huffman_tree, received_message_end)
print("Dec: {}".format(t.toc()))

decoded_message = np.array(decoded_message)
img.from_bitmap(decoded_message)
img.show()
