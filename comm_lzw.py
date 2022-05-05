import os
from io import StringIO

# if using anaconda3 and error execute: conda install --channel conda-forge pillow=5.2.0
import numpy as np

import lzw
import util
from channel import channel
from imageSource import ImageSource
from unireedsolomon import rs
from util import Time
import math

# ========================= SOURCE =========================
IMG_NAME = 'image.jpg'

dir_path = os.path.dirname(os.path.realpath(__file__))
IMG_PATH = os.path.join(dir_path, IMG_NAME)  # use absolute path

print(F"Loading {IMG_NAME} at {IMG_PATH}")
img = ImageSource().load_from_file(IMG_PATH)
print(img)

# ================================================================

t = Time()

# ======================= SOURCE ENCODING ========================
# ====================== Lempel-Ziv-Welch ========================

pixels = img.get_pixel_seq()
t.tic()
encoded_message_lzw, dictionary = lzw.encode(pixels)
print(max(encoded_message_lzw))
print("Enc: {}".format(t.toc()))

encoded_message_lzw_uint16 = np.array(encoded_message_lzw, dtype=np.uint16)
encoded_message_bit = util.uint16_to_bit(encoded_message_lzw_uint16)
print("Length message ", len(encoded_message_bit))

# ====================== CHANNEL ENCODING ========================
# ======================== Reed-Solomon ==========================

# as we are working with symbols of 8 bits
# choose n such that m is divisible by 8 when n=2^m−1
# Example: 255 + 1 = 2^m -> m = 8
n = 255  # code_word_length in symbols
k = 223  # message_length in symbols

coder = rs.RSCoder(n, k)

# calculate desired length of the message
length_message = 8*223*math.ceil(len(encoded_message_bit) / (8 * 223))

# calculate amount of padding
amount_padding = length_message - len(encoded_message_bit)

# add padding
encoded_message_bit = encoded_message_bit.zfill(length_message)

# convert from bits to bytes
encoded_message_uint8 = util.bit_to_uint8(encoded_message_bit)

# convert bytes to string
encoded_message_str = ""
for element in encoded_message_uint8:
    encoded_message_str += chr(element)

# divide string in blocks for channel encoding
messages = [encoded_message_str[idx: idx + k] for idx in range(0, len(encoded_message_str), 223)]

rs_encoded_message = StringIO()
t.tic()
for message in messages:
    code = coder.encode_fast(message, return_string=True)
    rs_encoded_message.write(code)

rs_encoded_message_uint8 = np.array(
    [ord(c) for c in rs_encoded_message.getvalue()], dtype=np.uint8)
print(t.toc())
print("ENCODING COMPLETE")

# convert to bits to send over channel
rs_encoded_message_bit = util.uint8_to_bit(rs_encoded_message_uint8)

# ========================== CHANNEL =============================

t.tic()
received_message = channel(rs_encoded_message_bit, ber=0.001)
t.toc_print()

# ====================== CHANNEL ENCODING ========================
# ======================== Reed-Solomon ==========================

# convert back to bytes and split in blocks
received_message_uint8 = util.bit_to_uint8(received_message)
received_message_uint8 = np.split(received_message_uint8, len(messages))

# split source encoded message to compare
rs_encoded_message_uint8 = np.split(rs_encoded_message_uint8, len(messages))

decoded_message = StringIO()

t.tic()
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

# read received message as byte stream
received_message_uint8 = np.array(
    [ord(c) for c in decoded_message.getvalue()], dtype=np.uint8)

# convert bytes to bits to remove padding
received_message_end = util.uint8_to_bit(received_message_uint8)
received_message_end = received_message_end[amount_padding:]

# ======================= SOURCE DECODING ========================
# ====================== Lempel-Ziv-Welch ========================

# convert bits to uint16 to decode
received_message_end = util.bit_to_uint16(received_message_end)

t.tic()
# decode
decoded_message = lzw.decode(received_message_end.tolist())
print("Enc: {0:.4f}".format(t.toc()))

# convert to numpy array for check and image
decoded_message = np.array(decoded_message)

# check if received pixels are equal to pixels from image
check = np.array_equal(pixels, decoded_message)
print("Same pixel sequence:", check)
