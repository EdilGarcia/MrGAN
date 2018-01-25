import sys
import os
import program

if sys.version_info[0] >= 3:
    import tkinter as tk
    import tkinter.filedialog as fl
    import tkinter.messagebox as msg
    import tkinter.font as font
else:
    import Tkinter as tk
    import tkFileDialog as fl
    import tkMessageBox as msg
    import tkFont as font

class mainMenu(tk.Toplevel):
    model_dir=''

    def open_settings(self):
        #Settings Window
        settings = tk.Tk()
        width = 260
        height = 270
        screenwidth = settings.winfo_screenwidth()
        screenheight = settings.winfo_screenheight()
        settings.title("MrGAN Settings")
        settings.geometry(
            '%dx%d+%d+%d' % (width, height, ((screenwidth / 2) - (width / 2)), ((screenheight / 2) - (height / 2))))
        settings.resizable(width=False, height=False)

        settings_frame = tk.Frame(settings, bg="#C4E0FE")
        settings_frame.pack(fill=tk.BOTH)

        label_epoch = tk.Label(settings_frame, text='Epoch:', bg="#C4E0FE")
        label_epoch.grid(row=0, column=0, sticky=tk.W)
        label_lr = tk.Label(settings_frame, text='Learning Rate:', bg="#C4E0FE")
        label_lr.grid(row=1, column=0, sticky=tk.W)
        label_batch = tk.Label(settings_frame, text='Batch Size:', bg="#C4E0FE")
        label_batch.grid(row=2, column=0, sticky=tk.W, rowspan=2)
        label_nnets = tk.Label(settings_frame, text='Number of\nNetworks:', bg="#C4E0FE")
        label_nnets.grid(row=4, column=0, sticky=tk.W)
        label_randave = tk.Label(settings_frame, text='Random Number\nAverage Times:', bg="#C4E0FE")
        label_randave.grid(row=5, column=0, sticky=tk.W, rowspan=2)
        label_randthrsh = tk.Label(settings_frame, text='Random Number\nThreshold:', bg="#C4E0FE")
        label_randthrsh.grid(row=7, column=0, sticky=tk.W, rowspan=2)
        label_model = tk.Label(settings_frame, text='Model Iterations\nSaver:', bg="#C4E0FE")
        label_model.grid(row=9, column=0, sticky=tk.W, rowspan=2)
        label_summary = tk.Label(settings_frame, text='Summary and sample\nImage output:', bg="#C4E0FE")
        label_summary.grid(row=11, column=0, sticky=tk.W, rowspan=2)
        #variables
        EPOCH = tk.Entry(settings_frame, width=20)
        EPOCH.grid(row=0, column=1)
        EPOCH.insert(0, "101")
        LEARNING_RATE = tk.Entry(settings_frame, width=20)
        LEARNING_RATE.grid(row=1, column=1)
        LEARNING_RATE.insert(0, "0.0002")
        BATCH_SIZE =  tk.Entry(settings_frame, width=20)
        BATCH_SIZE.grid(row=2, column=1)
        BATCH_SIZE.insert(0, "64")
        NUM_NETS = tk.Entry(settings_frame, width=20)
        NUM_NETS.grid(row=4, column=1)
        NUM_NETS.insert(0, "3")
        NUM_RAND_AVE = tk.Entry(settings_frame, width=20)
        NUM_RAND_AVE.grid(row=5, column=1)
        NUM_RAND_AVE.insert(0, "5")
        NUM_RAND_THRSH = tk.Entry(settings_frame, width=20)
        NUM_RAND_THRSH.grid(row=7, column=1)
        NUM_RAND_THRSH.insert(0, "0.9")
        MODEL_ITERS = tk.Entry(settings_frame, width=20)
        MODEL_ITERS.grid(row=9, column=1)
        MODEL_ITERS.insert(0, "5")
        SUMMARY_ITERS = tk.Entry(settings_frame, width=20)
        SUMMARY_ITERS.grid(row=11, column=1)
        SUMMARY_ITERS.insert(0, "5")
        entries = [EPOCH, LEARNING_RATE, BATCH_SIZE, NUM_NETS, NUM_RAND_AVE, NUM_RAND_AVE, MODEL_ITERS, SUMMARY_ITERS]
        img_apply_settings = tk.PhotoImage(file="./source/btn-start-01.png")
        btn_apply = tk.Button(settings_frame, text='apply', padx=15, command=lambda: self.apply_settings(entries,settings))
        btn_apply.grid(row=13, column=0, rowspan=2)

        img_cancel_settings = tk.PhotoImage(file="./source/btn-exit.png")
        btn_cancel = tk.Button(settings_frame, text='cancel', padx=15, command=lambda: settings.destroy())
        btn_cancel.grid(row=13, column=1, rowspan=2)

    def apply_settings(self, entries, settings):
        keys, value = ['EPOCH','LEARNING_RATE','BATCH_SIZE','NUM_NETS','MODEL_ITERS','SUMMARY_ITERS',
                      'NUM_RAND_AVE','NUM_RAND_THRSH'], [int(entries[0].get()), float(entries[1].get()),
                      int(entries[2].get()), int(entries[3].get()), int(entries[4].get()),
                      int(entries[5].get()), int(entries[6].get()), float(entries[7].get())]
        settings_dict = dict(zip(keys, value))
        program.set_params(settings_dict)
        settings.destroy()
    def start(self):
        self.destroy()
        self.original_frame.show()

    def exit_app(self):
        exit()

    def train(self):
        print('Commence training')
        program.train_mrgan()

    def generate(self):
        print('Generating Image')
        global set_dir
        if directory != '':
            program.generate()
        else:
            self.text_box.insert(tk.INSERT, "\n\n!!!INVALID MODEL!!!\n\n")

    def open_dir(self):
        global model_dir
        directory = fl.askdirectory()
        model_dir = directory
        program.set_folders(model_dir)
        self.set_image()

    def set_image(self):
        if model_dir != '':
            set_name = model_dir.split('/')[-1]
            sample = ''
            sample = os.listdir('./output/training_image_output/'+set_name)[0]
        if sample != '':
            photo = tk.PhotoImage(file="./source/placeholder.png")
        else:
            photo = ImageTk.PhotoImage(file="./source/placeholder.png")

        self.image_out_lbl.configure(image = photo)
        self.image_out_lbl.image = photo

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
        self.button_home = tk.Button(self, image=self.image_home, command=lambda: self.start())
        self.button_home.place(x=0, y=80, width=140, height=40)

        self.image_open = tk.PhotoImage(file="./source/btn-open.png")
        self.button_open = tk.Button(self, image=self.image_open, command=lambda: self.open_dir())
        self.button_open.place(x=140, y=80, width=170, height=40)

        self.image_train = tk.PhotoImage(file="./source/btn-train.png")
        self.button_train = tk.Button(self, image=self.image_train, command=lambda: self.train())
        self.button_train.place(x=310, y=80, width=180, height=40)

        self.image_generate = tk.PhotoImage(file="./source/btn-generate.png")
        self.button_generate = tk.Button(self, image=self.image_generate, command=lambda: self.generate())
        self.button_generate.place(x=490, y=80, width=170, height=40)

        self.image_exit = tk.PhotoImage(file="./source/btn-exit.png")
        self.button_exit = tk.Button(self, image=self.image_exit, command=lambda: self.exit_app())
        self.button_exit.place(x=660, y=80, width=140, height=40)

        #main frame
        self.main_frame = tk.Frame(self, width=800, height=380)
        self.main_frame.place(x=0, y=120)

        self.image_frame = tk.PhotoImage(file="./source/window-frame.png")
        self.label_frame = tk.Label(self.main_frame, image=self.image_frame)
        self.label_frame.place(x=0, y=0, width=800, height=380)

        #Logs
        self.text_frame_master = tk.Frame(self.main_frame)
        self.text_frame_master.place(x=0, y=0, width=310, height=380)

        self.text_frame = tk.Frame(self.text_frame_master)
        self.text_frame.pack(fill=tk.BOTH, expand=True)

        self.text_box = tk.Text(self.text_frame, bg="light sky blue", relief="sunken", width=32, height=20)
        self.text_box.config(font=("Arial", 12), undo=True, wrap='word')
        self.text_box.insert(tk.INSERT, "You need to finish your thesis.")
        self.text_box.pack(side=tk.LEFT, fill=tk.BOTH)

        self.scroll = tk.Scrollbar(self.text_frame)
        self.scroll.config(command = self.text_box.yview)
        self.text_box.config(yscrollcommand = self.scroll.set)
        self.scroll.pack(side=tk.RIGHT, fill=tk.Y)

        #Output images
        self.image_out_frame = tk.Frame(self.main_frame, bg="#C4E0FE")
        self.image_out_frame.place(x=310, y=0, width=500, height=380)

        self.image_settings = tk.PhotoImage(file="./source/btn-settings.png")
        self.button_settings = tk.Button(self.image_out_frame, image=self.image_settings, command=lambda: self.open_settings())
        self.button_settings.place(x=0, y=0, width=120, height=26)

        self.image_out_lbl = tk.Label(self.image_out_frame)
        self.image_out_lbl.place(x=10, y=40, width=470, height=320)
        self.set_image()

        program.init_objects(self.text_box, tk)

class startMenu(object):
    def about(self, master):
        self.master = master
        width = 800
        height = 500
        screenwidth = master.winfo_screenwidth()
        screenheight = master.winfo_screenheight()
        master.title("ABOUT")
        master.geometry(
            '%dx%d+%d+%d' % (width, height, ((screenwidth / 2) - (width / 2)), ((screenheight / 2) - (height / 2))))
        master.resizable(width=False, height=False)

        self.image_entire = tk.PhotoImage(file="./source/about-frame.png")
        self.label_entire = tk.Label(master, image=self.image_entire)
        self.label_entire.place(x=0, y=0, width=800, height=500)

        self.image_home = tk.PhotoImage(file="./source/btn-home-01.png")
        self.button_home = tk.Button(master, image=self.image_home, borderwidth=0, command=lambda: self.__init__(master))
        self.button_home.place(x=0, y=0, width=200, height=40)

        self.image_about = tk.PhotoImage(file="./source/btn-about-01.png")
        self.button_about = tk.Button(root, image=self.image_about, borderwidth=0, command=lambda: self.about(master))
        self.button_about.place(x=200, y=0, width=200, height=40)

        self.image_start = tk.PhotoImage(file="./source/btn-start-01.png")
        self.button_start = tk.Button(master, image=self.image_start, borderwidth=0, command=lambda: self.open_main())
        self.button_start.place(x=400, y=0, width=200, height=40)

        self.image_tips = tk.PhotoImage(file="./source/btn-tips-01.png")
        self.button_tips = tk.Button(master, image=self.image_tips, borderwidth=0, command=lambda: self.tips(master))
        self.button_tips.place(x=600, y=0, width=200, height=40)

        #with open("./source/about.txt", "r") as content_file:
        #    content = content_file.read()
        #msg.showinfo('About',content)

    def tips(self, master):
        self.master = master
        width = 800
        height = 500
        screenwidth = master.winfo_screenwidth()
        screenheight = master.winfo_screenheight()
        master.title("ABOUT")
        master.geometry(
            '%dx%d+%d+%d' % (width, height, ((screenwidth / 2) - (width / 2)), ((screenheight / 2) - (height / 2))))
        master.resizable(width=False, height=False)

        self.image_entire = tk.PhotoImage(file="./source/tips-frame.png")
        self.label_entire = tk.Label(master, image=self.image_entire)
        self.label_entire.place(x=0, y=0, width=800, height=500)

        self.image_home = tk.PhotoImage(file="./source/btn-home-02.png")
        self.button_home = tk.Button(master, image=self.image_home, borderwidth=0, command=lambda: self.__init__(master))
        self.button_home.place(x=0, y=0, width=200, height=40)

        self.image_about = tk.PhotoImage(file="./source/btn-about-02.png")
        self.button_about = tk.Button(root, image=self.image_about, borderwidth=0, command=lambda: self.about(master))
        self.button_about.place(x=200, y=0, width=200, height=40)

        self.image_start = tk.PhotoImage(file="./source/btn-start-01.png")
        self.button_start = tk.Button(master, image=self.image_start, borderwidth=0, command=lambda: self.open_main())
        self.button_start.place(x=400, y=0, width=200, height=40)

        self.image_tips = tk.PhotoImage(file="./source/btn-tips-02.png")
        self.button_tips = tk.Button(master, image=self.image_tips, borderwidth=0, command=lambda: self.tips(master))
        self.button_tips.place(x=600, y=0, width=200, height=40)

        #with open("./source/tips.txt", "r") as content_file:
        #    content = content_file.read()
        #msg.showinfo('About',content)

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
        self.button_about =  tk.Button(master, image=self.image_about, command=lambda:self.about(master))
        self.button_about.place(x=136, y=400, width=140, height=40)

        self.image_start =  tk.PhotoImage(file="./source/btn-start.png")
        self.button_start =  tk.Button(root, image=self.image_start, command=lambda:self.open_main())
        self.button_start.place(x=329, y=400, width=140, height=40)

        self.image_tips =  tk.PhotoImage(file="./source/btn-tips.png")
        self.button_tips =  tk.Button(master, image=self.image_tips, command=lambda:self.tips(master))
        self.button_tips.place(x=525, y=400, width=140, height=40)

if __name__ == "__main__":
    root = tk.Tk()
    start_window = startMenu(root)
    root.mainloop()
