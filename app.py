import tkinter as tk

import cv2
from PIL import Image, ImageTk
import os
from client.homepage import HomePage
from client.classpage import ClassPage
class WebcamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Webcam Viewer")


        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Không thể mở webcam")
            self.root.destroy()
            return


        self.label = tk.Label(self.root)
        self.label.pack()

        # Cập nhật khung hình
        self.update_frame()


        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame,1)
            img = ImageTk.PhotoImage(Image.fromarray(frame))
            self.label.imgtk = img
            self.label.configure(image=img)
        self.root.after(10, self.update_frame)

    def on_closing(self):
        self.cap.release()
        self.root.destroy()

class MenuBar:
    def __init__(self,root):
        self.root = root
        self.menu_bar_color = "#383838"
        self.page_frame = tk.Frame(self.root)
        self.page_frame.tk.call("source", "forest-light.tcl")
        self.page_frame.place(relwidth =1.0,relheight=1.0,x =50)
        self.menu_bar_frame = tk.Frame(self.root,bg=self.menu_bar_color)
        self.menu_bar_frame.pack(side=tk.LEFT, fill=tk.Y,pady =4,padx=4)
        self.menu_bar_frame.pack_propagate(flag=False)
        self.menu_bar_frame.config(width=45)

        self.toggle_icon = tk.PhotoImage(file ="images/toggle_btn_icon.png")
        self.home_icon = tk.PhotoImage(file ="images/home_icon.png")
        self.class_manage = tk.PhotoImage(file="images/services_icon.png")
        self.setting_icon = tk.PhotoImage(file="images/updates_icon.png")
        self.close_toggle = tk.PhotoImage(file="images/close_btn_icon.png")

        self.button_toggle = tk.Button(self.menu_bar_frame,image=self.toggle_icon,bg=self.menu_bar_color,bd=0,activebackground=self.menu_bar_color,command=self.extend_menubar)
        self.button_toggle.place(x=4,y=10)
        self.button_home = tk.Button(self.menu_bar_frame,image=self.home_icon,bg=self.menu_bar_color,bd=0,activebackground=self.menu_bar_color,command=lambda:self.switch_indicator(self.home_indicator,self.home_page))
        self.button_home.place(x=9,y=130,width = 30, height =40)
        self.button_class_manage = tk.Button(self.menu_bar_frame,image=self.class_manage,bg=self.menu_bar_color,bd=0,activebackground=self.menu_bar_color,command=lambda:self.switch_indicator(self.class_manage_indicator,self.class_manage_page))
        self.button_class_manage.place(x=9,y=190, width = 30, height =40)
        self.button_setting = tk.Button(self.menu_bar_frame,image=self.setting_icon,bg=self.menu_bar_color,bd=0,activebackground=self.menu_bar_color,command=lambda:self.switch_indicator(self.setting_indicator,self.setting_page))
        self.button_setting.place(x=9,y=250, width = 30, height =40)

        self.home_indicator = self.btn_indicator(y =130)
        self.setting_indicator = self.btn_indicator(y =250)
        self.class_manage_indicator = self.btn_indicator(y =190)

        self.home_label = tk.Label(self.menu_bar_frame,text="Home",bg=self.menu_bar_color,fg="white", font=("Times",15))
        self.home_label.place(x=45,y=130,width = 100, height =40)
        self.class_manage_label = tk.Label(self.menu_bar_frame,text="Class",bg=self.menu_bar_color,fg="white", font=("Times",15))
        self.class_manage_label.place(x=45,y=190,width = 100, height =40)
        self.setting_label = tk.Label(self.menu_bar_frame,text="Setting",bg=self.menu_bar_color,fg="white", font=("Times",15))
        self.setting_label.place(x=45,y=250,width = 100, height =40)

        self.home_label.bind("<Button-1>",lambda e:self.switch_indicator(self.home_indicator,self.home_page))
        self.class_manage_label.bind("<Button-1>",lambda e:self.switch_indicator(self.class_manage_indicator,self.class_manage_page))
        self.setting_label.bind("<Button-1>",lambda e:self.switch_indicator(self.setting_indicator,self.setting_page))


    def btn_indicator(self,y):
        btn = tk.Label(self.menu_bar_frame,bg = self.menu_bar_color)
        btn.place(x=3,y=y,width=3 ,height=40)
        return btn
    def switch_indicator(self,indicator,page):
        self.home_indicator.config(bg=self.menu_bar_color)
        self.setting_indicator.config(bg=self.menu_bar_color)
        self.class_manage_indicator.config(bg=self.menu_bar_color)
        indicator.config(bg="white")
        if self.menu_bar_frame.winfo_width() >45:
            self.close_menubar()
        for child in self.page_frame.winfo_children():
            child.destroy()
        page()
    def extend_menubar(self):
        self.extending_animation()
        self.button_toggle.config(image=self.close_toggle,command=self.close_menubar)

    def close_menubar(self):
        self.close_animation()
        self.button_toggle.config(image=self.toggle_icon,command=self.extend_menubar)

    def extending_animation(self):
        current_width = self.menu_bar_frame.winfo_width()
        if current_width < 200:
            self.menu_bar_frame.config(width=current_width+10)
            self.root.after(1,self.extending_animation)
    def close_animation(self):
        current_width = self.menu_bar_frame.winfo_width()
        if current_width > 45:
            self.menu_bar_frame.config(width=current_width-10)
            self.root.after(1,self.close_animation)




    def class_manage_page(self):
        self.class_manage_page_frame = ClassPage(self.page_frame)

    def home_page(self):
        self.home_page_frame = HomePage(self.page_frame)

        self.home_page_frame.pack(fill = tk.BOTH,expand=True)
    def setting_page(self):
        self.setting_page_frame = tk.Frame(self.page_frame,bg="green")
        self.setting_page_frame.pack(fill = tk.BOTH,expand=True)
        lb=tk.Label(self.setting_page_frame,text="Setting Page",font=("Arial",20))
        lb.pack()

if __name__=="__main__":
    root = tk.Tk()
    root.geometry("1280x720")
    menu_bar = MenuBar(root)

    menu_bar.root.mainloop()