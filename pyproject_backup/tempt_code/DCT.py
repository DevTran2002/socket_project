import numpy as np
from scipy.fftpack import dct, idct
import cv2
import sys
import heapq
from collections import defaultdict


quantization_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                [12, 12, 14, 19, 26, 58, 60, 55],
                                [14, 13, 16, 24, 40, 57, 69, 56],
                                [14, 17, 22, 29, 51, 87, 80, 62],
                                [18, 22, 37, 56, 68, 109, 103, 77],
                                [24, 35, 55, 64, 81, 104, 113, 92],
                                [49, 64, 78, 87, 103, 121, 120, 101],
                                [72, 92, 95, 98, 112, 100, 103, 99]])


def rgb_to_ycbcr(img):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128

    return y, cb, cr

def dct_2d(block):
    return dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct_2d(block):
    return idct(idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

def quantize_macroblocks(macroblocks, quantization_factor):
    return (macroblocks // quantization_factor).astype(np.uint8)

# Hàm để giải lượng tử hóa các macroblocks
def dequantize_macroblocks(quantized_macroblocks, quantization_factor):
    return quantized_macroblocks * quantization_factor
#
# def dct_2d(block):
#     result = np.zeros_like(block, dtype=float)
#
#     for u in range(8):
#         for v in range(8):
#             cu = 1.0 / np.sqrt(2) if u == 0 else 1.0
#             cv = 1.0 / np.sqrt(2) if v == 0 else 1.0
#
#             sum_value = 0.0
#             for x in range(8):
#                 for y in range(8):
#                     sum_value += block[x, y] * np.cos((2 * x + 1) * u * np.pi / 16) * np.cos((2 * y + 1) * v * np.pi / 16)
#
#             result[u, v] = 0.25 * cu * cv * sum_value
#
#     return result
#
#
# def idct_2d(dct_block):
#     result = np.zeros_like(dct_block, dtype=float)
#
#     for x in range(8):
#         for y in range(8):
#             sum_value = 0.0
#             for u in range(8):
#                 for v in range(8):
#                     cu = 1.0 / np.sqrt(2) if u == 0 else 1.0
#                     cv = 1.0 / np.sqrt(2) if v == 0 else 1.0
#
#                     sum_value += cu * cv * dct_block[u, v] * np.cos((2 * x + 1) * u * np.pi / 16) * np.cos((2 * y + 1) * v * np.pi / 16)
#
#             result[x, y] = 0.25 * sum_value
#
#     return result

def resize_with_padding(image, block_size):
    # Lấy kích thước ảnh
    height, width, channels = image.shape

    # Tính toán số lượng pixel cần thêm vào để ảnh chia hết cho block_size
    padding_height = (height // block_size + 1) * block_size - height
    padding_width = (width // block_size + 1) * block_size - width

    # Thêm padding vào ảnh
    if(padding_height != 0 and padding_width !=0):
        padded_image = np.pad(image, ((0, padding_height), (0, padding_width), (0, 0)), mode='constant')

    return padded_image


def apply_dct(img):
    block_size = 8
    height, width, _ = img.shape

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = img[i:i+block_size, j:j+block_size, :]
            y, cb, cr = rgb_to_ycbcr(block)

            y_dct = dct_2d(y)
            cb_dct = dct_2d(cb)
            cr_dct = dct_2d(cr)

            print(y_dct.shape)
            # **Hàm lượng tử hóa**
            y_dct_quantized = quantize_macroblocks(y_dct, quantization_matrix)
            cb_dct_quantized = quantize_macroblocks(cb_dct, quantization_matrix)
            cr_dct_quantized = quantize_macroblocks(cr_dct, quantization_matrix)

            # # Perform quantization here (not shown in this example)
            # y_idct = idct_2d(y_dct)
            # cb_idct = idct_2d(cb_dct)
            # cr_idct = idct_2d(cr_dct)
            #
            # Update the original block with the inverse DCT results
            img[i:i + block_size, j:j + block_size, 0] = y_dct_quantized
            img[i:i + block_size, j:j + block_size, 1] = cb_dct_quantized
            img[i:i + block_size, j:j + block_size, 2] = cr_dct_quantized

    return img

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
            loadImg(frame)

            # Bấm q để thoát
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()

    # đóng hết các frame
    cv2.destroyAllWindows()

def loadImg(img):
    # Load an RGB image
    block_size = 8
    # image_path = 'hinhhoc.jpg'
    # _image = cv2.imread(image_path)
    _image =img
    original_image = resize_with_padding(_image, block_size)

    # Apply DCT to the image
    compressed_image = apply_dct(original_image)

    # Mã hóa độ dài  (RLC) cho từng macroblock
    rle_encoded_macroblocks = [run_length_encode(macroblock.flatten()) for macroblock in compressed_image]
    # Hiển thị thông tin về dữ liệu đã mã hóa RLE
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
    print("Huffman Mapping:", huffman_mapping)

    # Áp dụng bảng mã hóa cho tất cả các macroblocks
    huffman_encoded_macroblocks = [
        [huffman_mapping[symbol] for symbol, _ in rle_data]
        for rle_data in rle_encoded_macroblocks
    ]

    sizeof = sys.getsizeof(huffman_encoded_macroblocks)

    print("huffman encode", sizeof)
    cv2.imshow('Compressed Image', compressed_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

loadVideo()
