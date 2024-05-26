import tkinter as tk
import random

# Constants
GRID_SIZE = 80
CELL_SIZE = 6.5  # Size of each cell in pixels

# Create the main window
root = tk.Tk()
root.title("80x80 Random Grid")

# Create a canvas widget
canvas = tk.Canvas(root, width=GRID_SIZE * CELL_SIZE+200, height=GRID_SIZE * CELL_SIZE+200)
canvas.pack()

# Function to draw the grid
def draw_grid():
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x1 = col * CELL_SIZE +100
            y1 = row * CELL_SIZE + 75
            x2 = x1 + CELL_SIZE
            y2 = y1 + CELL_SIZE
            color = random.choice(["black", "white"])
            canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")


# Function to draw the zebra head
def draw_zebra_head():
    # Head is a simple oval
    head_x1 = (GRID_SIZE  - 5) * CELL_SIZE +100
    head_y1 = 10
    head_x2 = (GRID_SIZE + 5) * CELL_SIZE +100
    head_y2 = 10 * CELL_SIZE +10
    canvas.create_oval(head_x1, head_y1, head_x2, head_y2, fill="white", outline="black")

    # Eyes
    eye_size = 2 * CELL_SIZE
    left_eye_x1 = head_x1 + 2 * CELL_SIZE
    left_eye_y1 = head_y1 + 2 * CELL_SIZE
    left_eye_x2 = left_eye_x1 + eye_size
    left_eye_y2 = left_eye_y1 + eye_size

    right_eye_x1 = head_x2 - 4 * CELL_SIZE
    right_eye_y1 = head_y1 + 2 * CELL_SIZE
    right_eye_x2 = right_eye_x1 + eye_size
    right_eye_y2 = right_eye_y1 + eye_size

    canvas.create_oval(left_eye_x1, left_eye_y1, left_eye_x2, left_eye_y2, fill="black", outline="black")
    canvas.create_oval(right_eye_x1, right_eye_y1, right_eye_x2, right_eye_y2, fill="black", outline="black")

# Function to draw the zebra legs
def draw_zebra_legs():
    leg_width = 3 * CELL_SIZE
    leg_height = 5 * CELL_SIZE
    leg_positions = [
        (GRID_SIZE // 4, GRID_SIZE + 11),
        (GRID_SIZE // 4 + 15, GRID_SIZE + 11  ),
        (3 * GRID_SIZE // 4 - 15, GRID_SIZE + 11  ),
        (3 * GRID_SIZE // 4, GRID_SIZE +11)
    ]
    for pos in leg_positions:
        x1 = pos[0] * CELL_SIZE
        y1 = pos[1] * CELL_SIZE
        x2 = x1 + leg_width
        y2 = y1 + leg_height
        canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="black")

# Function to draw the zebra tail
def draw_zebra_tail():
    tail_x1 = 25
    tail_y1 = (GRID_SIZE // 2) * CELL_SIZE
    tail_x2 =  10 * CELL_SIZE +25
    tail_y2 = tail_y1 + 1 * CELL_SIZE
    canvas.create_line(tail_x1, tail_y1, tail_x2, tail_y2, fill="black", width=5)

# Draw the zebra body, head, legs, and tail
draw_grid()
draw_zebra_head()
draw_zebra_legs()
draw_zebra_tail()



# Run the Tkinter event loop
root.mainloop()
