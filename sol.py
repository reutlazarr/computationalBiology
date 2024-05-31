import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import random

# Constants
GRID_SIZE = 80
CELL_SIZE = 6.5  # Size of each cell in pixels

class CellularAutomaton:
    def __init__(self, size=80):
        self.size = size
        self.grid = np.random.choice([0, 1], size=(size, size))
        self.next_grid = np.copy(self.grid)

    def update(self):
        for i in range(self.size):
            for j in range(self.size):
                self.next_grid[i, j] = self.rule(i, j)
        self.grid, self.next_grid = self.next_grid, self.grid

    def rule(self, x, y):
        left = self.grid[x, (y-1) % self.size]
        leftAbove = self.grid[(x-1) % self.size, (y-1) % self.size]
        leftBelow = self.grid[(x+1) % self.size, (y-1) % self.size]
        right = self.grid[x, (y+1) % self.size]
        rightAbove = self.grid[(x-1) % self.size, (y+1) % self.size]
        rightBelow = self.grid[(x+1) % self.size, (y+1) % self.size]
        above = self.grid[(x-1) % self.size, y]
        below = self.grid[(x+1) % self.size, y]
        threshold = 0
        me = self.grid[x][y]
        if me == above:
            threshold += 1/8
        if me == below:
            threshold += 1/8  
        if me != left:
            threshold += 1/8  
        if me != right:
            threshold += 1/8 
        if me != rightAbove:
            threshold += 1/8
        if me != rightBelow:
            threshold += 1/8 
        if me != leftAbove:
            threshold += 1/8
        if me != leftBelow:
            threshold += 1/8
        if threshold == 0.5:
            return random.choice([0, 1])
        if threshold < 0.5:
            return 1 - me
        return me

    def stripe_index(self):
        mismatches = 0
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.grid[i, j] == self.grid[i + 1, j]:
                    mismatches += 1
        return mismatches / (self.size * (self.size - 1))

    def zebra_index(self):
        score = 0
        for j in range(self.size):
            white_ratio = np.sum(self.grid[:, j]) / self.size  # Ratio of white cells in the column
            dominant_color = 1 if white_ratio > 0.5 else 0  # 1 if most cells are white, 0 if most are black

            # Neighboring columns
            left_col = self.grid[:, (j-1) % self.size]
            right_col = self.grid[:, (j+1) % self.size]

            left_ratio = np.sum(left_col) / self.size  # Ratio of white cells in the left column
            right_ratio = np.sum(right_col) / self.size  # Ratio of white cells in the right column

            left_color = 1 if left_ratio > 0.5 else 0
            right_color = 1 if right_ratio > 0.5 else 0

            if left_color != dominant_color and right_color != dominant_color:
                score += white_ratio if dominant_color == 1 else (1 - white_ratio)

        return score / self.size  # Normalize by the size of the matrix

# Create the main window
root = tk.Tk()
root.title("80x80 Cellular Automaton")

# Create a canvas widget
canvas = tk.Canvas(root, width=GRID_SIZE * CELL_SIZE + 200, height=GRID_SIZE * CELL_SIZE + 200)
canvas.pack()

# Initialize the cellular automaton
ca = CellularAutomaton(GRID_SIZE)
stripe_indices = []
zebra_indices = []
generation = 0

def draw_grid():
    canvas.delete("all")
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x1 = col * CELL_SIZE + 100
            y1 = row * CELL_SIZE + 75
            x2 = x1 + CELL_SIZE
            y2 = y1 + CELL_SIZE
            color = "black" if ca.grid[row, col] == 1 else "white"
            canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
            
    draw_generation_number()
    draw_zebra_head()
    draw_zebra_legs()
    draw_zebra_tail()

def draw_zebra_head():
    # Head is a simple oval
    head_x1 = (GRID_SIZE  - 5) * CELL_SIZE + 100
    head_y1 = 10
    head_x2 = (GRID_SIZE + 5) * CELL_SIZE + 100
    head_y2 = 10 * CELL_SIZE + 10
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

def draw_zebra_legs():
    leg_width = 3 * CELL_SIZE
    leg_height = 5 * CELL_SIZE
    leg_positions = [
        (GRID_SIZE // 4, GRID_SIZE + 11),
        (GRID_SIZE // 4 + 15, GRID_SIZE + 11),
        (3 * GRID_SIZE // 4 - 15, GRID_SIZE + 11),
        (3 * GRID_SIZE // 4, GRID_SIZE + 11)
    ]
    for pos in leg_positions:
        x1 = pos[0] * CELL_SIZE
        y1 = pos[1] * CELL_SIZE
        x2 = x1 + leg_width
        y2 = y1 + leg_height
        canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="black")

def draw_zebra_tail():
    tail_x1 = 25
    tail_y1 = (GRID_SIZE // 2) * CELL_SIZE
    tail_x2 = 10 * CELL_SIZE + 25
    tail_y2 = tail_y1 + 1 * CELL_SIZE
    canvas.create_line(tail_x1, tail_y1, tail_x2, tail_y2, fill="black", width=5)

def plot_zebra_index():
    plt.plot(zebra_indices, color='tab:red')
    plt.xlabel('Generation')
    plt.ylabel('Zebra Index')
    plt.title('Zebra Index Over Generations')
    plt.ylim(0, 1)  # Ensure the y-axis ranges from 0 to 1
    plt.show()

def update():
    global generation
    if generation < 400:
        ca.update()
        draw_grid()
        zebra_indices.append(ca.zebra_index())
        generation += 1
        root.after(100, update)
    else:
        plot_zebra_index()

def draw_generation_number():
    # Display the current generation number
    zebra_index_value = ca.zebra_index()
    canvas.create_text(50, 20, text=f"Generation: {generation}" , anchor="nw", font=("Helvetica", 16))
    canvas.create_text(50, 40, text=f"Zebra Index: {zebra_index_value:.2f}", anchor="nw", font=("Helvetica", 16))

# Draw the initial grid
draw_grid()

# Start the update loop
root.after(100, update)

# Run the Tkinter event loop
root.mainloop()
