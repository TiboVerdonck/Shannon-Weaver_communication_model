# TODO Install necessary packages via: conda install --file requirements.txt

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
import random
import string

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

# Use t.tic() and t.toc() to measure the executing time as shown below

t = Time()

t.tic()
# TODO Determine the number of occurrences of the source or use a fixed huffman_freq
pixels = img.get_pixel_seq()
huffman_freq = Counter(pixels).items()
huffman_tree = huffman.Tree(huffman_freq)
print(F"Generating the Huffman Tree took {t.toc_str()}")

t.tic()
# TODO print-out the codebook and validate the codebook (include your findings in the report)
print(huffman_tree.codebook)
encoded_message = huffman.encode(huffman_tree.codebook, img.get_pixel_seq())
print(len(encoded_message))
print("Enc huffman: {}".format(t.toc()))

t.tic()
decoded_message = huffman.decode(huffman_tree, encoded_message)

print("Dec huffman: {}".format(t.toc()))

input_lzw = img.get_pixel_seq().copy()

# ======================= SOURCE ENCODING ========================
# ====================== Lempel-Ziv-Welch ========================

t.tic()
encoded_msg, dictonary = lzw.encode(input_lzw)
print("Enc lzw: {}".format(t.toc()))

t.tic()
decoded_msg = lzw.decode(encoded_msg)
print("Dec lzw: {0:.4f}".format(t.toc()))

uint8_stream = np.array(decoded_msg, dtype=np.uint8)
# ====================== CHANNEL ENCODING ========================
# ======================== Reed-Solomon ==========================

# as we are working with symbols of 8 bits
# choose n such that m is divisible by 8 when n=2^m???1
# Example: 255 + 1 = 2^m -> m = 8
n = 255  # code_word_length in symbols
k = 223  # message_length in symbols

coder = rs.RSCoder(n, k)

# TODO generate a matrix with k symbols per rows (for each message)
# TODO afterwards you can iterate over each row to encode the message

messages = []

for x in range(0, 10):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(k))
    messages.append(result_str)

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
received_message = channel(rs_encoded_message_bit, ber=0.55)
t.toc_print()

# TODO Use this helper function to convert a bit stream to a uint8 stream
received_message_uint8 = util.bit_to_uint8(received_message)
received_message_uint8 = np.split(received_message_uint8, len(messages))

rs_encoded_message_uint8 = np.split(rs_encoded_message_uint8, len(messages))

decoded_message = StringIO()

t.tic()


# TODO Iterate over the received messages and compare with the original RS-encoded messages
for cnt, (block, original_block) in enumerate(zip(received_message_uint8, rs_encoded_message_uint8)):
    try:
        decoded, ecc = coder.decode_fast(block, return_string=True)
        assert coder.check(decoded + ecc), "Check not correct"
        decoded_message.write(str(decoded))
    except rs.RSCodecError as error:
        diff_symbols = len(block) - (original_block == block).sum()
        print(
            F"Error occured after {cnt} iterations of {len(received_message_uint8)}")
        print(F"{diff_symbols} different symbols in this block")

t.toc_print()

print("DECODING COMPLETE")

# TODO after everything works, try to simulate the communication model as specified in the assignment

