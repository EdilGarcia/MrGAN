import sys
import os
import program

if sys.version_info[0] >= 3:
    import tkinter as tk
    import tkinter.filedialog as fl
    import tkinter.messagebox as msg
else:
    import Tkinter as tk
    import tkFileDialog as fl
    import tkMessageBox as msg

class mainMenu(tk.Toplevel):
    model_dir=None

    def close_window(self):
        self.destroy()
        self.original_frame.show()

    def exit_app(self):
        exit()

    def train(self):
        print('Commence training')
        program.train_mrgan()

    def generate(self):
        print('Generating Image')
        global model_dir
        if model_dir is not None:
            program.generate(model_dir)
        else:
            self.text_box.insert(tk.INSERT, "\n\n!!!INVALID MODEL!!!\n\n")

    def open_dir(self):
        global model_dir
        directory = fl.askdirectory()
        model_dir = directory
        valid = program.set_folders(model_dir)
        if valid == 0:
            msg.showinfo('Error','Invalid Directory!')

    def __init__(self, original):
        self.original_frame = original
        tk.Toplevel.__init__(self)
        width = 800
        height = 500
        screenwidth = self.winfo_screenwidth()
        screenheight = self.winfo_screenheight()
        self.title("MAIN MENU")
        self.geometry(
            '%dx%d+%d+%d' % (width, height, ((screenwidth / 2) - (width / 2)), ((screenheight / 2) - (height / 2))))
        self.resizable(width=False, height=False)

        # entire frame
        self.image_entire = tk.PhotoImage(file='./source/entire-main-frame.png')
        self.label_entire = tk.Label(self, image=self.image_entire)
        self.label_entire.place(x=0, y=0, width=800, height=500)

        # buttons
        self.image_home = tk.PhotoImage(file="./source/btn-home.png")
        self.button_home = tk.Button(self, image=self.image_home, command=lambda: self.close_window())
        self.button_home.place(x=0, y=80, width=100, height=40)

        self.image_open = tk.PhotoImage(file="./source/btn-open.png")
        self.button_open = tk.Button(self, image=self.image_open, command=lambda: self.open_dir())
        self.button_open.place(x=100, y=80, width=200, height=40)

        self.image_train = tk.PhotoImage(file="./source/btn-train.png")
        self.button_train = tk.Button(self, image=self.image_train, command=lambda: self.train())
        self.button_train.place(x=300, y=80, width=200, height=40)

        self.image_generate = tk.PhotoImage(file="./source/btn-generate.png")
        self.button_generate = tk.Button(self, image=self.image_generate, command=lambda: self.generate())
        self.button_generate.place(x=500, y=80, width=200, height=40)

        self.image_exit = tk.PhotoImage(file="./source/btn-exit.png")
        self.button_exit = tk.Button(self, image=self.image_exit, command=lambda: self.exit_app())
        self.button_exit.place(x=700, y=80, width=100, height=40)

        #main frame
        self.main_frame = tk.Frame(self, width=800, height=380)
        self.main_frame.place(x=0, y=120)

        self.image_frame = tk.PhotoImage(file="./source/window-frame.png")
        self.label_frame = tk.Label(self.main_frame, image=self.image_frame)
        self.label_frame.place(x=0, y=0, width=800, height=380)

        #Logs
        self.text_frame = tk.Frame(self.main_frame)
        self.text_frame.pack()

        self.text_box = tk.Text(self.text_frame, bg="grey", relief="sunken", width=25, height=21)
        self.text_box.config(font=("Courier New", 12), undo=True, wrap='word')
        self.text_box.insert(tk.INSERT, "Hello.....")
        self.text_box.pack(side=tk.LEFT, fill=tk.BOTH)

        self.scroll = tk.Scrollbar(self.text_frame)
        self.scroll.config(command = self.text_box.yview)
        self.text_box.config(yscrollcommand = self.scroll.set)
        self.scroll.pack(side=tk.RIGHT, fill=tk.Y)

        #Output images
        self.image_out_frame = tk.Frame(self.main_frame, bg="blue")
        self.text_frame.pack()

        program.init_objects(self.text_box, tk)

class startMenu(object):
    def about(self):
        with open("./source/about.txt", "r") as content_file:
            content = content_file.read()
        msg.showinfo('About',content)

    def tips(self):
        with open("./source/tips.txt", "r") as content_file:
            content = content_file.read()
        msg.showinfo('About',content)

    def open_main(self):
        self.master.withdraw()
        main_window = mainMenu(self)

    def show(self):
        self.master.update()
        self.master.deiconify()

    def __init__(self, master):

        self.master = master

        width = 800
        height = 500
        screenwidth = master.winfo_screenwidth()
        screenheight = master.winfo_screenheight()
        master.title("START MENU")
        master.geometry('%dx%d+%d+%d' % (width, height, ((screenwidth/2) - (width/2)), ((screenheight/2) - (height/2))))
        master.resizable(width=False, height=False)

        #entire frame
        self.image_entire =  tk.PhotoImage(file="./source/entire-frame.png")
        self.label_entire =  tk.Label(master, image=self.image_entire)
        self.label_entire.place(x=0, y=0, width=800, height=500)

        #buttons
        self.image_about =  tk.PhotoImage(file="./source/btn-about.png")
        self.button_about =  tk.Button(master, image=self.image_about, command=lambda:self.about())
        self.button_about.place(x=136, y=400, width=140, height=40)

        self.image_start =  tk.PhotoImage(file="./source/btn-start.png")
        self.button_start =  tk.Button(root, image=self.image_start, command=lambda:self.open_main())
        self.button_start.place(x=329, y=400, width=140, height=40)

        self.image_tips =  tk.PhotoImage(file="./source/btn-tips.png")
        self.button_tips =  tk.Button(master, image=self.image_tips, command=lambda:self.tips())
        self.button_tips.place(x=525, y=400, width=140, height=40)

if __name__ == "__main__":
    root = tk.Tk()
    start_window = startMenu(root)
    root.mainloop()
