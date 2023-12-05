from numpy import *
import cv2
import numpy as np
import math
from scipy.fftpack import dct, idct
from PIL import Image
import heapq
from collections import defaultdict
import time
import os
import sys
quantization_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                [12, 12, 14, 19, 26, 58, 60, 55],
                                [14, 13, 16, 24, 40, 57, 69, 56],
                                [14, 17, 22, 29, 51, 87, 80, 62],
                                [18, 22, 37, 56, 68, 109, 103, 77],
                                [24, 35, 55, 64, 81, 104, 113, 92],
                                [49, 64, 78, 87, 103, 121, 120, 101],
                                [72, 92, 95, 98, 112, 100, 103, 99]])


def dct_2d(block):
    result = np.zeros_like(block, dtype=float)

    for u in range(8):
        for v in range(8):
            cu = 1.0 / np.sqrt(2) if u == 0 else 1.0
            cv = 1.0 / np.sqrt(2) if v == 0 else 1.0

            sum_value = 0.0
            for x in range(8):
                for y in range(8):
                    sum_value += block[x, y] * np.cos((2 * x + 1) * u * np.pi / 16) * np.cos((2 * y + 1) * v * np.pi / 16)

            result[u, v] = 0.25 * cu * cv * sum_value

    return result


def idct_2d(dct_block):
    result = np.zeros_like(dct_block, dtype=float)

    for x in range(8):
        for y in range(8):
            sum_value = 0.0
            for u in range(8):
                for v in range(8):
                    cu = 1.0 / np.sqrt(2) if u == 0 else 1.0
                    cv = 1.0 / np.sqrt(2) if v == 0 else 1.0

                    sum_value += cu * cv * dct_block[u, v] * np.cos((2 * x + 1) * u * np.pi / 16) * np.cos((2 * y + 1) * v * np.pi / 16)

            result[x, y] = 0.25 * sum_value

    return result

def resize_with_padding(image, block_size):
    # Lấy kích thước ảnh
    height, width, channels = image.shape

    # Tính toán số lượng pixel cần thêm vào để ảnh chia hết cho block_size
    padding_height = (height // block_size + 1) * block_size - height
    padding_width = (width // block_size + 1) * block_size - width

    # Thêm padding vào ảnh
    padded_image = np.pad(image, ((0, padding_height), (0, padding_width), (0, 0)), mode='constant')

    return padded_image




def split_into_macroblocks(image, block_size):
    # Kích thước của ảnh
    image_height, image_width, _ = image.shape

    # Số lượng macroblocks theo chiều cao và chiều rộng
    num_macroblocks_height = image_height // block_size
    num_macroblocks_width = image_width // block_size

    # Tách ảnh thành các macroblock 8x8
    macroblocks = []
    for i in range(num_macroblocks_height):
        for j in range(num_macroblocks_width):
            macroblock = image[i * block_size: (i + 1) * block_size,
                         j * block_size: (j + 1) * block_size, :]
            macroblocks.append(macroblock)

    # Chuyển đổi danh sách các macroblocks thành một mảng NumPy
    macroblocks_array = np.array(macroblocks)

    return macroblocks_array

def combine_macroblocks(macroblocks, image_shape):
    image_height, image_width, _ = image_shape
    macroblock_size = macroblocks.shape[1]
    num_macroblocks_height = image_height // macroblock_size
    num_macroblocks_width = image_width // macroblock_size

    combined_image = np.zeros(image_shape, dtype=np.uint8)

    for i in range(num_macroblocks_height):
        for j in range(num_macroblocks_width):
            combined_image[i * macroblock_size : (i + 1) * macroblock_size,
                           j * macroblock_size : (j + 1) * macroblock_size, :] = macroblocks[i * num_macroblocks_width + j]

    return combined_image


def quantize_macroblocks(macroblocks, quantization_factor):
    return (macroblocks // quantization_factor).astype(np.uint8) * quantization_factor

# Hàm để giải lượng tử hóa các macroblocks
def dequantize_macroblocks(quantized_macroblocks, quantization_factor):
    return quantized_macroblocks * quantization_factor

def run_length_encode(data):
    encoded_data = []
    count = 1
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            count += 1
        else:
            encoded_data.append((data[i - 1], count))
            count = 1
    encoded_data.append((data[-1], count))
    return encoded_data


class HuffmanNode:
    def __init__(self, symbol=None, frequency=None):
        self.symbol = symbol
        self.frequency = frequency
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.frequency < other.frequency

def build_huffman_tree(frequencies):
    heap = [HuffmanNode(symbol=s, frequency=f) for s, f in frequencies.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)

        merged_node = HuffmanNode(frequency=node1.frequency + node2.frequency)
        merged_node.left = node1
        merged_node.right = node2

        heapq.heappush(heap, merged_node)

    return heap[0]

def build_huffman_codes(node, code="", mapping=None):
    if mapping is None:
        mapping = {}

    if node.symbol is not None:
        mapping[node.symbol] = code
    if node.left is not None:
        build_huffman_codes(node.left, code + "0", mapping)
    if node.right is not None:
        build_huffman_codes(node.right, code + "1", mapping)

    return mapping

def huffman_encode_block(block, huffman_mapping):
    encoded_block = ""
    for value in block.flatten():
        encoded_block += huffman_mapping[value]
    return encoded_block

def huffman_decode(encoded_data, huffman_tree):
    decoded_data = []
    current_node = huffman_tree

    for bit in encoded_data:
        if bit == '0':
            current_node = current_node.left
        elif bit == '1':
            current_node = current_node.right



        if current_node.symbol is not None:
            decoded_data.append(current_node.symbol)
            current_node = huffman_tree

    return decoded_data

def loadVideo():

    cap = cv2.VideoCapture(r"C:\Users\Public\Documents\PythonCode\myvideo.mjpeg")
    # Kiểm tra videoo được mở không
    if (cap.isOpened() == False):
        print("Error opening video file")

        # Đọc video
    while (cap.isOpened()):

        # Tách từng khung hình
        ret, frame = cap.read()

        if ret == True:

            # cv2.imshow('Frame', frame)
            compress(frame)

            # Bấm q để thoát
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break


        else:
            break


    cap.release()

    # đóng hết các frame
    cv2.destroyAllWindows()


def compress(img):


    real_time = time.time()
    mb_size = 8
    # Thực hiện resize với padding
    resized_image = resize_with_padding(img, mb_size)
    # /////////////////////////////////////////////////////
    # Chuyển đổi ảnh thành kiểu dữ liệu float32
    image_float = resized_image.astype(np.float32)
    # image_float = resized_image
    compressed_size1 = sys.getsizeof(image_float)
    print(compressed_size1)

    # Tách các kênh màu
    red_channel = resized_image[:, :, 0]
    green_channel = resized_image[:, :, 1]
    blue_channel = resized_image[:, :, 2]
    # Áp dụng DCT cho từng kênh màu
    red_dct = dct(dct(red_channel, axis=0, norm='ortho'), axis=1, norm='ortho')
    green_dct = dct(dct(green_channel, axis=0, norm='ortho'), axis=1, norm='ortho')
    blue_dct = dct(dct(blue_channel, axis=0, norm='ortho'), axis=1, norm='ortho')

    # Gộp các kênh màu để có ảnh ban đầu
    reconstructed_image = np.stack([red_dct, green_dct, blue_dct], axis=2).clip(0, 255).astype(np.uint8)
    macroblocks = split_into_macroblocks(reconstructed_image, mb_size)

    for i in range(macroblocks.shape[0]):
        # Lượng tử hóa các macroblocks
        quantization_factor = 2
        quantized_macroblocks = quantize_macroblocks(macroblocks, quantization_factor)

    # Mã hóa độ dài  (RLC) cho từng macroblock
    rle_encoded_macroblocks = [run_length_encode(macroblock.flatten()) for macroblock in quantized_macroblocks]

    # # Hiển thị thông tin về dữ liệu đã mã hóa RLE
    # for i, rle_data in enumerate(rle_encoded_macroblocks):
    #     print(f"RLE Encoded Data for Macroblock {i + 1}:", rle_data)

    # Tính toán tần số xuất hiện từ dữ liệu RLE
    frequencies = defaultdict(int)
    for rle_data in rle_encoded_macroblocks:
        for symbol, count in rle_data:
            frequencies[symbol] += count


    # Xây dựng cây Huffman từ tần số xuất hiện
    huffman_tree = build_huffman_tree(frequencies)

    # Xây dựng bảng mã hóa từ cây Huffman
    huffman_mapping = build_huffman_codes(huffman_tree)

    # Hiển thị bảng mã hóa
    # print("Huffman Mapping:", huffman_mapping)

    # Áp dụng bảng mã hóa cho tất cả các macroblocks
    huffman_encoded_macroblocks = [
        [huffman_mapping[symbol] for symbol, _ in rle_data]
        for rle_data in rle_encoded_macroblocks
    ]
    real_time1 = time.time()

    print(real_time1-real_time)


    compressed_size = sys.getsizeof(huffman_encoded_macroblocks)
    compressed_size_2 = sys.getsizeof(rle_encoded_macroblocks)
    print(rle_encoded_macroblocks)
    for i in range(macroblocks.shape[0]):
        # Giải lượng tử hóa các macroblocks
        dequantized_macroblocks = dequantize_macroblocks(quantized_macroblocks, quantization_factor)

    # # Áp dụng ngược DCT để khôi phục ảnh ban đầu từ các kênh đã áp dụng DCT
    # red_idct = idct(idct(red_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
    # green_idct = idct(idct(green_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
    # blue_idct = idct(idct(blue_dct, axis=0, norm='ortho'), axis=1, norm='ortho')

    # Gộp các kênh màu để có ảnh ban đầu
    # reconstructed_image = np.stack([red_idct, green_idct, blue_idct], axis=2).clip(0, 255).astype(np.uint8)


loadVideo()

