import tkinter as tk
import random
import numpy as np



# Create the main window


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
        neighbors = [self.grid[(x-1)%self.size, y], self.grid[(x+1)%self.size, y],
                     self.grid[x, (y-1)%self.size], self.grid[x, (y+1)%self.size]]
        if sum(neighbors) < 2:
            return 1
        elif sum(neighbors) > 2:
            return 0
        else:
            return random.choice([0, 1])

    def stripe_index(self):
        mismatches = 0
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.grid[i, j] == self.grid[i, j + 1]:
                    mismatches += 1
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.grid[i, j] == self.grid[i + 1, j]:
                    mismatches += 1
        return mismatches / (2 * self.size * (self.size - 1))
    

# Constants
GRID_SIZE = 80
CELL_SIZE = 6.5  # Size of each cell in pixels
class App:
    def __init__(self, root):
        self.root = root
        self.size = 80
        self.ca = CellularAutomaton(self.size)
        self.canvas = tk.Canvas(root, width=self.size*10+200, height=self.size*10+200)
        self.canvas.pack()
        self.running = False
        self.generation = 0
        self.stripe_indices = []

        self.run_button = tk.Button(root, text="Run", command=self.run)
        self.run_button.pack()
    



    # Function to draw the grid
    def draw_grid(self):
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x1 = col * CELL_SIZE +100
                y1 = row * CELL_SIZE + 75
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE
                color = random.choice(["black", "white"])
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
        self.canvas.update()


    # Function to draw the zebra head
    def draw_zebra_head(self):
        # Head is a simple oval
        head_x1 = (GRID_SIZE  - 5) * CELL_SIZE +100
        head_y1 = 10
        head_x2 = (GRID_SIZE + 5) * CELL_SIZE +100
        head_y2 = 10 * CELL_SIZE +10
        self.canvas.create_oval(head_x1, head_y1, head_x2, head_y2, fill="white", outline="black")

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

        self.canvas.create_oval(left_eye_x1, left_eye_y1, left_eye_x2, left_eye_y2, fill="black", outline="black")
        self.canvas.create_oval(right_eye_x1, right_eye_y1, right_eye_x2, right_eye_y2, fill="black", outline="black")

    # Function to draw the zebra legs
    def draw_zebra_legs(self):
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
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="black")

    # Function to draw the zebra tail
    def draw_zebra_tail(self):
        tail_x1 = 25
        tail_y1 = (GRID_SIZE // 2) * CELL_SIZE
        tail_x2 =  10 * CELL_SIZE +25
        tail_y2 = tail_y1 + 1 * CELL_SIZE
        self.canvas.create_line(tail_x1, tail_y1, tail_x2, tail_y2, fill="black", width=5)

    # Initialize the canvas with the zebra drawing
    def run(self):
        self.draw_grid()
        self.draw_zebra_head()
        self.draw_zebra_legs()
        self.draw_zebra_tail()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Zebra")
    app = App(root)
    root.mainloop()