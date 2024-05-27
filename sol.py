import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt

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
        left = self.grid[x, (y-1)%self.size]
        right = self.grid[x, (y+1)%self.size]
        above = self.grid[(x-1)%self.size, y]
        below = self.grid[(x+1)%self.size, y]
        
        if left == right:
            return 1 - left
        elif above == below:
            return 1 - above
        else:
            return self.grid[x, y]

    def stripe_index(self):
        mismatches = 0
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.grid[i, j] == self.grid[i + 1, j]:
                    mismatches += 1
        return mismatches / (self.size * (self.size - 1))

# Create the main window
root = tk.Tk()
root.title("80x80 Cellular Automaton")

# Create a canvas widget
canvas = tk.Canvas(root, width=GRID_SIZE * CELL_SIZE + 200, height=GRID_SIZE * CELL_SIZE + 200)
canvas.pack()

# Initialize the cellular automaton
ca = CellularAutomaton(GRID_SIZE)
stripe_indices = []
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

    draw_zebra_head()
    draw_zebra_legs()
    draw_zebra_tail()

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

def draw_zebra_tail():
    tail_x1 = 25
    tail_y1 = (GRID_SIZE // 2) * CELL_SIZE
    tail_x2 =  10 * CELL_SIZE +25
    tail_y2 = tail_y1 + 1 * CELL_SIZE
    canvas.create_line(tail_x1, tail_y1, tail_x2, tail_y2, fill="black", width=5)

def update():
    global generation
    if generation < 250:
        ca.update()
        draw_grid()
        stripe_indices.append(ca.stripe_index())
        generation += 1
        root.after(100, update)
    else:
        plot_stripe_indices()

def plot_stripe_indices():
    plt.plot(stripe_indices)
    plt.xlabel('Generation')
    plt.ylabel('Stripe Index')
    plt.title('Stripe Index Over Generations')
    plt.show()

# Draw the initial grid
draw_grid()

# Start the update loop
root.after(100, update)

# Run the Tkinter event loop
root.mainloop()