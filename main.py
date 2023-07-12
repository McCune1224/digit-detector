import tkinter as tk
from PIL import Image
import numpy as np
import pandas as pd

# Library for separating actual Random Forest Classifier from GUI
import handwriting_lib

# CONSTANTS SETTINGS FOR GUI ELEMENTS
# ====================================================================================================

# GUI Window 
# TODO make this resizable by a flag and allow user to resize window
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600



BLACK_PIXEL = np.full((1, 3), 0, dtype=np.uint8)
WHITE_PIXEL = np.full((1, 3), 255, dtype=np.uint8)
# Canvas length should scale with Window Width and Height but keep it divisible by 28
CANVAS_LENGTH = 140 
CANVAS_GRID = np.full((CANVAS_LENGTH, CANVAS_LENGTH, 3), BLACK_PIXEL, dtype=np.uint8) # Ideally should scale to the same size as the mnist dataset (28x28)



#Colors
CANVAS_BACKGROUND_COLOR = "#AFAFAF" 
BUTTON_BACKGROUND_COLOR = "#EFEFEF"
BUTTON_FOREGROUND_COLOR = "#000000"



# ====================================================================================================


def _canvas_grid_to_1d_df(user_drawing_matrix: np.ndarray):
    """
    Helper for downscaling the 140x140 pixel grid the user gave 
    down to a dataframe of 784 columns comparable to that of the mnist dataset.

    Args:
        user_drawing_array (np.ndarray): 140x140 matrix of pixels representation of 
        the user's drawing.

    Returns: 
        single column DataFrame of length 784 with a value of either 0 (white) or 1 (black)
    """

    # List to convert into mnlist_df
    mnist_list = []

    # TODO
    # 1. Downscale drawing array dimensions from 140x140 to 28x28
    # currently using a janky one liner to shrink array down by factor of 5 (which is okay since 140/5 = 28)
    downscaled_drawing_matrix = user_drawing_matrix[::5, ::5]

    # 2. flatten drawing array 28x28 array into 1d array
    flattened_array = downscaled_drawing_matrix.ravel()

    # 3. Enumerate the drawing array, and based off every 3rd index,
    # append the inverted color (0 to our mnist list if value is 255 (white) or 1 if
    # the value 0 (black))
    for idx in range(0, len(flattened_array), 3):
        if flattened_array[idx] == 255:
            mnist_list.append(255)
        else:
            mnist_list.append(0)

    # 4. Convert to Dataframe
    print(mnist_list)
    mnist_df = pd.DataFrame(columns=list(range(len(mnist_list))))
    # One Giant Row with 784 columns, each column representation an rgb value of either 0 or 255
    mnist_df.loc[0] = mnist_list

    # Writing to file as well just for viewing and debugging
    mnist_df.to_csv("user_drawing_28.csv", sep='\t')
    # Can't use 1d array for this, so just use the downscaled_drawing_matrix made before-hand
    pimage = Image.fromarray(downscaled_drawing_matrix)
    pimage.save('user_drawing_28.jpg')

    return mnist_df


def predict_drawing(rfc, canvas):
    """
    Predicts the digit drawn by a user using a trained Random Forest Classifier based off
    mnist dataset.

    """
    user_mnist_df = _canvas_grid_to_1d_df(canvas)

    drawing_prediction = rfc.predict(user_mnist_df)

    # Hard to see without having to rescale tkinter GUI, so printing to terminal as well for debugging
    print(f"Prediction is {drawing_prediction[0]}")
    label4.configure(text=f"Prediction is {drawing_prediction[0]}")


def draw_handwriting(event):
    ''' Draws on the canvas and internal array based on mouse coordinates. '''

    # xy-coordinates and radius of the circle being drawn
    x = event.x
    y = event.y
    r = 2
    cvs_drawspace.create_oval(x - r, y - r, x + r+1, y + r+1, fill='black')
    CANVAS_GRID[y-2:y+3, x-2:x+3] = WHITE_PIXEL


def clear_drawing():
    ''' Clears the drawing canvas and internal array. '''

    cvs_drawspace.delete('all')  # drawing canvas
    CANVAS_GRID[:] = BLACK_PIXEL  # internal representation of drawing


if __name__ == "__main__":

    # https://www.askpython.com/python-modules/tkinter/python-tkinter-grid-example
    # Model Prep before running our GUI

    # Model Creation and Accuracy Score
    # ====================================================================================================
    handwriting_model, accuracy_score = handwriting_lib.create_rfc_model(
        'mnist_train.csv')
    # ====================================================================================================

    # GUI Creation
    # ====================================================================================================

    # Create the master object
    window = tk.Tk()
    window.columnconfigure(0, minsize=300)
    window.rowconfigure([0, 1], minsize=200)
    window.configure(bg="#534D56")
    window.focusmodel

    # hard coded accuracy for now while debugging


    #Top of GUI displays accuracy score
    label1 = tk.Label(
        text=f"RFC Accuracy: {round((100*accuracy_score), 2)}%", bg=BUTTON_BACKGROUND_COLOR, fg="#000000")
    label1.grid(row=0, column=0, sticky="n")

    # Button for calling the predict_drawing function
    # had to put in Lambda since by default this acts more of a delegate which means 
    # no adding arguments to our function in 'command=func', but lambda lets me work around that :)
    label2 = tk.Button(text="Submit", command=lambda: predict_drawing(
        handwriting_model, CANVAS_GRID), bg=BUTTON_BACKGROUND_COLOR, foreground=BUTTON_FOREGROUND_COLOR)
    label2.grid(row=1, column=0, sticky="n")

    cvs_drawspace = tk.Canvas(width=CANVAS_LENGTH, height=CANVAS_LENGTH, bg='white', cursor='tcross',
                              highlightthickness=1, highlightbackground='orange')
    cvs_drawspace.grid(row=2, column=0, sticky="n")
    cvs_drawspace.bind('<B1-Motion>', draw_handwriting)
    cvs_drawspace.bind('<Button-1>', draw_handwriting)


    label3 = tk.Button(text="Clear Canvas", command=clear_drawing,
                       bg=BUTTON_BACKGROUND_COLOR, foreground=BUTTON_FOREGROUND_COLOR)
    label3.grid(row=3, column=0, sticky="n")

    label4 = tk.Label(
        text=f"Prediction: ", bg=BUTTON_BACKGROUND_COLOR, fg=BUTTON_FOREGROUND_COLOR)
    label4.grid(row=1, column=2, sticky="n")

    window.resizable(False, False)
    window.mainloop()
# ==================================================
