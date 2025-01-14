import tkinter as tk
from tkinter import ttk
from  models import *
class HomePage(tk.Frame):
    def __init__(self, root):
        # Gọi __init__ của lớp cha (tk.Frame) để khởi tạo các thuộc tính của Frame
        super().__init__(root)

        self.root = root  # Đặt root là đối tượng cha của HomePage

        label = tk.Label(self, text="Welcome to the Home Page", font=("Arial", 24))
        label.pack(pady=50)

        button = tk.Button(self, text="Go to Settings", command=self.go_to_settings)
        button.pack(pady=20)

    def go_to_settings(self):
        # Chuyển đến trang settings
        self.root.show_page("settings")