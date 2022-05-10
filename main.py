import math
import tkinter as tk
from tkinter import filedialog as fd
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageTk


# A Huffman Tree Node
class Node:
    def __init__(self, prob, symbol, left=None, right=None):
        # probability of symbol
        self.prob = prob

        # symbol
        self.symbol = symbol

        # left node
        self.left = left

        # right node
        self.right = right

        # tree direction (0/1)
        self.code = ''


""" A helper function to print the codes of symbols by traveling Huffman Tree"""
codes = dict()


def Calculate_Codes(node, val=''):
    # huffman code for current node
    newVal = val + str(node.code)

    if node.left:
        Calculate_Codes(node.left, newVal)
    if node.right:
        Calculate_Codes(node.right, newVal)

    if not node.left and not node.right:
        codes[node.symbol] = newVal

    return codes


""" A helper function to calculate the probabilities of symbols in given data"""


def calculateProbability(data):
    symbols_dict = dict()
    for element in data:
        if symbols_dict.get(element) is None:
            symbols_dict[element] = 1
        else:
            symbols_dict[element] += 1
    return symbols_dict


""" A helper function to obtain the encoded output"""


def Output_Encoded(data, coding):
    encoding_output = []
    for c in data:
        encoding_output.append(coding[c])
    string = ''.join([str(item) for item in encoding_output])
    return string


""" A helper function to calculate the space difference between compressed and non compressed data"""


def Total_Gain(data, coding):
    global before_compression
    global after_compression
    before_compression = len(data) * 8  # total bit space to stor the data before compression
    after_compression = 0
    symbols = coding.keys()
    for symbol in symbols:
        count = data.count(symbol)
        after_compression += count * len(coding[symbol])  # calculate how many bit is required for that symbol in total
    print("Space usage before compression (in bits):", before_compression)
    print("Space usage after compression (in bits):", after_compression)
    print("Compression ratio is:", before_compression / after_compression)


def Huffman_Encoding(data):
    global entropy
    global averageCodeLength
    symbol_with_probs = calculateProbability(data)
    symbols = symbol_with_probs.keys()
    probabilities = symbol_with_probs.values()
    print("symbols:", symbols)
    print("probabilities:", probabilities)

    nodes = []

    # converting symbols and probabilities into huffman tree nodes
    for symbol in symbols:
        nodes.append(Node(symbol_with_probs.get(symbol), symbol))

    while len(nodes) > 1:
        # sort all the nodes in ascending order based on their probability
        nodes = sorted(nodes, key=lambda x: x.prob)
        # for node in nodes:
        #      print(node.symbol, node.prob)

        # pick 2 smallest nodes
        right = nodes[0]
        left = nodes[1]

        left.code = 0
        right.code = 1

        # combine the 2 smallest nodes to create new node
        newNode = Node(left.prob + right.prob, left.symbol + right.symbol, left, right)

        nodes.remove(left)
        nodes.remove(right)
        nodes.append(newNode)

    huffman_encoding = Calculate_Codes(nodes[0])
    totalProbability = 0
    averageCodeLength = 0
    entropy = 0
    for i in symbol_with_probs.values():
        totalProbability += i
    for i in range(len(probabilities)):
        averageCodeLength += (list(probabilities)[i] / totalProbability) * len(huffman_encoding.get(list(symbols)[i]))
        entropy += -((list(probabilities)[i] / totalProbability) * math.log((list(probabilities)[i] / totalProbability),
                                                                            2))
    print("Average Code Length:", averageCodeLength, "bits/symbol")
    print("Entropy:", entropy)

    print("symbols with codes", huffman_encoding)
    Total_Gain(data, huffman_encoding)
    encoded_output = Output_Encoded(data, huffman_encoding)
    return encoded_output, nodes[0]


def huffmanDecoding(encoded_data, huffman_tree, choose):
    tree_head = huffman_tree
    decoded_output = []
    for x in encoded_data:
        if x == '1':
            huffman_tree = huffman_tree.right
        elif x == '0':
            huffman_tree = huffman_tree.left
        if huffman_tree.left is None and huffman_tree.right is None:
            decoded_output.append(huffman_tree.symbol)
            huffman_tree = tree_head
    decoded_output = np.array(decoded_output)
    if choose == 5:
        string = ''.join([str(item) for item in decoded_output])
        return string
    elif choose == 0:
        decoded_output = decoded_output.reshape(gui_img.height(), gui_img.width())
    elif choose == 1:
        decoded_output = decoded_output.reshape(gui_img.height(), gui_img.width(), 3)
    return decoded_output


# Display the image in gray scale
def display_in_grayscale(image_panel):
    # open the current image as a PIL image
    img_rgb = Image.open(an_image_string)
    # convert the image to grayscale (img_grayscale has only one color channel
    # whereas img_rgb has 3 color channels as red, green and blue)
    img_grayscale = img_rgb.convert('L')
    print('\nFor the grayscale image')
    print('----------------------------------------------------------------------')
    img_grayscale_array = np.array(img_grayscale)
    print("the dimensions of the image array:", img_grayscale_array.shape)
    encoding, gray_tree = Huffman_Encoding(img_grayscale_array.tobytes())
    encodedOutput = open("Encoded text.txt", "w+")
    encodedOutput.truncate(1)
    encodedOutput.write(encoding)
    encodedOutput.close()
    # Show the decoded image
    decoded_photo = huffmanDecoding(encoding, gray_tree, 0)
    decoded_photo = Image.fromarray(np.uint8(decoded_photo))
    decoded_photo.save('decodedimage.jpg')
    img = ImageTk.PhotoImage(image=decoded_photo)
    image_panel.config(image=img)
    image_panel.photo_ref = img
    data = 'Entropy: {0}\nAverage Code Length: {1} bits/symbol\nCompression Ratio:{2}\nInput Image Size: {3} ' \
           'bits\nCompressed Image Size: {4} bits\nDifference: {5} bits\nDecoded photo saved as decodedimage.jpg'.format(
        str(entropy), str(averageCodeLength), str(before_compression / after_compression), str(before_compression),
        str(after_compression), str(before_compression - after_compression))
    text_panel = tk.Label(text_frame, text=data)
    text_panel.grid(row=10, column=0, columnspan=1, padx=15, pady=7)


# Display the colored image
def display_color_channel(image_panel, channel):
    # rgb -> no channel index, red channel -> 0, green channel -> 1 and blue channel -> 2
    # open the current image as a PIL image
    global decoded_photo
    img_rgb = Image.open(an_image_string)
    # convert the current image to a numpy array
    image_array = np.array(img_rgb)
    if channel == 'color':
        # Encode the image and save the encoded output
        encoding, color_tree = Huffman_Encoding(image_array.tobytes())
        encodedOutput = open("Encoded text.txt", "w+")
        encodedOutput.truncate(1)
        encodedOutput.write(encoding)
        encodedOutput.close()
        decoded_photo = huffmanDecoding(encoding, color_tree, 1)
        decoded_photo = Image.fromarray(np.uint8(decoded_photo))
        decoded_photo.save('decodedimage.jpg')
        img = ImageTk.PhotoImage(image=decoded_photo)
        image_panel.config(image=img)
        image_panel.photo_ref = img
        data = 'Entropy: {0}\nAverage Code Length: {1} bits/symbol\nCompression Ratio:{2}\nInput Image Size: {3} ' \
               'bits\nCompressed Image Size: {4} bits\nDifference: {5} bits\nDecoded photo saved as decodedimage.jpg\nPlease restart to choose a text.'.format(
            str(entropy), str(averageCodeLength), str(before_compression / after_compression), str(before_compression),
            str(after_compression), str(before_compression - after_compression))
        text_panel = tk.Label(text_frame, text=data)
        text_panel.grid(row=10, column=0, columnspan=1, padx=15, pady=7)
        return
    elif channel == 'red':
        channel_index = 0
    elif channel == 'green':
        channel_index = 1
    else:
        channel_index = 2
    # traverse all the pixels in the image array
    n_rows = image_array.shape[0]
    n_cols = image_array.shape[1]
    for row in range(n_rows):
        for col in range(n_cols):
            # make all the values 0 for the color channels except the given channel
            for rgb in range(3):
                if rgb != channel_index:
                    image_array[row][col][rgb] = 0
    if channel == 'red':
        # Encode the image and save the encoded output
        encoding, red_tree = Huffman_Encoding(image_array.tobytes())
        decoded_photo = huffmanDecoding(encoding, red_tree, 1)
    elif channel == 'green':
        # Encode the image and save the encoded output
        encoding, green_tree = Huffman_Encoding(image_array.tobytes())
        decoded_photo = huffmanDecoding(encoding, green_tree, 1)
    else:
        # Encode the image and save the encoded output
        encoding, blue_tree = Huffman_Encoding(image_array.tobytes())
        decoded_photo = huffmanDecoding(encoding, blue_tree, 1)
    encodedOutput = open("Encoded text.txt", "w+")
    encodedOutput.truncate(1)
    encodedOutput.write(encoding)
    encodedOutput.close()
    # convert the modified image array (numpy) to a PIL image
    decoded_photo = Image.fromarray(np.uint8(decoded_photo))
    # modify the displayed image
    decoded_photo.save('decodedimage.jpg')
    img = ImageTk.PhotoImage(image=decoded_photo)
    image_panel.config(image=img)
    image_panel.photo_ref = img
    data = 'Entropy: {0}\nAverage Code Length: {1} bits/symbol\nCompression Ratio:{2}\nInput Image Size: {3} ' \
           'bits\nCompressed Image Size: {4} bits\nDifference: {5} bits\nDecoded photo saved as decodedimage.jpg \nPlease restart to choose a text.'.format(
            str(entropy), str(averageCodeLength), str(before_compression / after_compression), str(before_compression),
            str(after_compression), str(before_compression - after_compression))
    text_panel = tk.Label(text_frame, text=data)
    text_panel.grid(row=10, column=0, columnspan=1, padx=15, pady=7)


def open_image():
    global gui_img
    global gui_img_panel
    global an_image_string
    global second_image_panel
    backup_an_image_string = an_image_string
    an_image_string = fd.askopenfilename(title='Select an image file', filetypes=[("image", ".jpeg"), ("image", ".png"), ("image", ".jpg"), ("image", ".bmp")])
    # display a warning message when the user does not select an image file
    if an_image_string == '':
        messagebox.showinfo('Warning', 'No image file is selected/opened.')
        an_image_string = backup_an_image_string
    # otherwise, modify the global variable image_file_path and the displayed image
    else:
        second_image_panel.destroy()
        second_image_panel = tk.Label(second_frame)
        second_image_panel.grid(row=0, column=7, columnspan=6, padx=10, pady=17)
        gui_img = ImageTk.PhotoImage(file=an_image_string)
        gui_img_panel.config(image=gui_img)
        gui_img_panel.photo_ref = gui_img
        print('The width in pixels:', gui_img.width(), ' The height in pixels:', gui_img.height())


def text(image_panel):
    file = open(fd.askopenfilename(title='Select an text file', filetypes=[("text", ".txt")]))
    context = file.read()
    if context == '':
        messagebox.showinfo('Warning', 'No file is selected/opened.')
    else:
        encoding, text_tree = Huffman_Encoding(context)
        encodedOutput = open("Encoded text.txt", "w+")
        encodedOutput.truncate(1)
        encodedOutput.write(encoding)
        encodedOutput.close()
        decoded_text = huffmanDecoding(encoding, text_tree, 5)
        decodedOutput = open("Decoded text.txt", "w+")
        decodedOutput.truncate(1)
        decodedOutput.write(decoded_text)
        decodedOutput.close()
        image_panel.config(None)
        image_panel.photo_ref = None
        image_panel.destroy()
        data = 'Entropy: {0}\nAverage Code Length: {1} bits/symbol\nCompression Ratio:{2}\nInput Text Size: {3} ' \
               'bits\nCompressed Text Size: {4} bits\nDifference: {5} bits\nDecoded text saved as Decoded text.txt\n ' \
               'Please restart to choose a new image.'.format(
                str(entropy), str(averageCodeLength), str(before_compression / after_compression), str(before_compression),
                str(after_compression), str(before_compression - after_compression))
        text_panel = tk.Label(text_frame, text=data)
        text_panel.grid(row=10, column=0, columnspan=1, padx=15, pady=7)


""" Main """
# create a window for the graphical user interface (gui)
gui = tk.Tk()
# set the title of the window
gui.title('Image Operations')
# set the background color of the window
gui['bg'] = 'SeaGreen1'
# create and place a frame on the window with some padding for all four sides
frame = tk.Frame(gui)
second_frame = tk.Frame(gui)
text_frame = tk.Frame(gui)
# using the grid method for layout management
frame.grid(row=0, column=0, padx=15, pady=15)
second_frame.grid(row=0, column=7, padx=15, pady=22)
text_frame.grid(row=10, column=3, padx=15, pady=7)
# set the background color of the frame
frame['bg'] = 'DodgerBlue4'
second_frame['bg'] = 'DodgerBlue4'
text_frame['bg'] = 'SeaGreen1'
# read and display the default image
an_image_string = fd.askopenfilename()
gui_img = ImageTk.PhotoImage(file=an_image_string)
gui_img_panel = tk.Label(frame, image=gui_img)
second_image_panel = tk.Label(second_frame)
# column span = 7 -> 7 columns as there are 7 buttons
gui_img_panel.grid(row=0, column=0, columnspan=7, padx=10, pady=10)
second_image_panel.grid(row=0, column=7, columnspan=1, padx=10, pady=17)
print('The width in pixels:', gui_img.width(), ' The height in pixels:', gui_img.height())
# create and place five buttons below the image (button commands are expressed
# as lambda functions for enabling input arguments)
# ----------------------------------------------------------------------------
# the first button enables the user to open and view an image from a file
btn1 = tk.Button(frame, text='Open Image', width=10)
btn1['command'] = lambda: open_image()
btn1.grid(row=1, column=0)
# create and place the second button that shows the image in grayscale
btn2 = tk.Button(frame, text='Grayscale', bg='gray', width=10)
btn2.grid(row=1, column=1)
# noinspection PyTypeChecker
btn2['command'] = lambda: display_in_grayscale(second_image_panel)
# create and place the third button that shows the red channel of the image
btn3 = tk.Button(frame, text='Red', bg='red', width=10)
btn3.grid(row=1, column=2)
# noinspection PyTypeChecker
btn3['command'] = lambda: display_color_channel(second_image_panel, 'red')
# create and place the third button that shows the green channel of the image
btn4 = tk.Button(frame, text='Green', bg='SpringGreen2', width=10)
btn4.grid(row=1, column=3)
# noinspection PyTypeChecker
btn4['command'] = lambda: display_color_channel(second_image_panel, 'green')
# create and place the third button that shows the blue channel of the image
btn5 = tk.Button(frame, text='Blue', bg='DodgerBlue2', width=10)
btn5.grid(row=1, column=4)
# noinspection PyTypeChecker
btn5['command'] = lambda: display_color_channel(second_image_panel, 'blue')
btn6 = tk.Button(frame, text='Color', bg='white', width=10)
# noinspection PyTypeChecker
btn6['command'] = lambda: display_color_channel(second_image_panel, 'color')
btn6.grid(row=1, column=5)
btn7 = tk.Button(frame, text='Text', bg='white', width=10)
# noinspection PyTypeChecker
btn7['command'] = lambda: text(gui_img_panel)
btn7.grid(row=1, column=6)
gui.mainloop()
