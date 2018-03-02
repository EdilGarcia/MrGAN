import sys
import os
import program
from PIL import Image, ImageTk
import glob
# import imageio
# imageio.plugins.ffmpeg.download()
import moviepy.editor as mpy

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

    def open_settings(self):
        #Settings Window
        settings = tk.Tk()

        width, height = 480, 279
        screenwidth = settings.winfo_screenwidth()
        screenheight = settings.winfo_screenheight()
        settings.title("MrGAN Settings")
        settings.geometry('%dx%d+%d+%d' % (width, height, ((screenwidth / 2) - (width / 2)), ((screenheight / 2) - (height / 2))))
        settings.resizable(width=False, height=False)

        settings_frame = tk.Frame(settings, bg="plum")
        settings_frame.pack(fill=tk.BOTH)

        label_epoch = tk.Label(settings_frame, text='Epoch:', font=20, bg="plum")
        label_epoch.grid(row=0, column=0, sticky=tk.W, pady=2)
        label_lr = tk.Label(settings_frame, text='Learning Rate:', font=20, bg="plum")
        label_lr.grid(row=1, column=0, sticky=tk.W, pady=2)
        label_batch = tk.Label(settings_frame, text='Batch Size:', font=20, bg="plum")
        label_batch.grid(row=2, column=0, sticky=tk.W, rowspan=2, pady=2)
        label_nnets = tk.Label(settings_frame, text='Number of Networks:', font=20, bg="plum")
        label_nnets.grid(row=4, column=0, sticky=tk.W, pady=2)
        label_randave = tk.Label(settings_frame, text='Random Number Average Times:',font=20, bg="plum")
        label_randave.grid(row=5, column=0, sticky=tk.W, rowspan=2, pady=2)
        label_randthrsh = tk.Label(settings_frame, text='Random Number Threshold:', font=20, bg="plum")
        label_randthrsh.grid(row=7, column=0, sticky=tk.W, rowspan=2, pady=2)
        label_model = tk.Label(settings_frame, text='Model Iterations Saver:', font=20, bg="plum")
        label_model.grid(row=9, column=0, sticky=tk.W, rowspan=2, pady=2)
        label_summary = tk.Label(settings_frame, text='Summary and Sample Image Output:', font=20, bg="plum")
        label_summary.grid(row=11, column=0, sticky=tk.W, rowspan=2, pady=2)
        label_netoworks = tk.Label(settings_frame, text='Architecture of Network:', font=20, bg="plum")
        label_netoworks.grid(row=13, column=0, sticky=tk.W, rowspan=2, pady=2)
        #variables
        EPOCH = tk.Entry(settings_frame, font=20, width=20)
        EPOCH.grid(row=0, column=1, pady=2)
        EPOCH.insert(0, '{}'.format(self.dict['EPOCH']))
        LEARNING_RATE = tk.Entry(settings_frame, font=20, width=20)
        LEARNING_RATE.grid(row=1, column=1, pady=2)
        LEARNING_RATE.insert(0, '{}'.format(self.dict['LEARNING_RATE']))
        BATCH_SIZE =  tk.Entry(settings_frame, font=20, width=20)
        BATCH_SIZE.grid(row=2, column=1, pady=2)
        BATCH_SIZE.insert(0, '{}'.format(self.dict['BATCH_SIZE']))
        NUM_NETS = tk.Entry(settings_frame, font=20, width=20)
        NUM_NETS.grid(row=4, column=1, pady=2)
        NUM_NETS.insert(0, '{}'.format(self.dict['NUM_NETS']))
        NUM_RAND_AVE = tk.Entry(settings_frame, font=20, width=20)
        NUM_RAND_AVE.grid(row=5, column=1, pady=2)
        NUM_RAND_AVE.insert(0, '{}'.format(self.dict['NUM_RAND_AVE']))
        NUM_RAND_THRSH = tk.Entry(settings_frame, font=20, width=20)
        NUM_RAND_THRSH.grid(row=7, column=1, pady=2)
        NUM_RAND_THRSH.insert(0, '{}'.format(self.dict['NUM_RAND_THRSH']))
        MODEL_ITERS = tk.Entry(settings_frame, font=20, width=20)
        MODEL_ITERS.grid(row=9, column=1, pady=2)
        MODEL_ITERS.insert(0, '{}'.format(self.dict['MODEL_ITERS']))
        SAVE_ITERS = tk.Entry(settings_frame, font=20, width=20)
        SAVE_ITERS.grid(row=11, column=1, pady=2)
        SAVE_ITERS.insert(0, '{}'.format(self.dict['SAVE_ITERS']))
        NETWORKS = tk.Entry(settings_frame, font=20, width=20)
        NETWORKS.grid(row=13, column=1, pady=2)
        NETWORKS.insert(0, '{}'.format(self.dict['NETWORKS']))
        entries = [EPOCH, LEARNING_RATE, BATCH_SIZE, NUM_NETS, NUM_RAND_AVE, NUM_RAND_AVE, MODEL_ITERS, SAVE_ITERS, NETWORKS]

        #img_apply_settings = tk.PhotoImage(file="./source/btn-start-01.png")
        btn_apply = tk.Button(settings_frame, text='Apply', padx=15, command=lambda: self.apply_settings(entries,settings))
        btn_apply.grid(row=15, column=0, rowspan=2)

        #img_cancel_settings = tk.PhotoImage(file="./source/btn-exit.png")
        btn_cancel = tk.Button(settings_frame, text='Cancel', padx=15, command=lambda: settings.destroy())
        btn_cancel.grid(row=15, column=1, rowspan=2)

    def apply_settings(self, entries, settings):
        keys, value = ['EPOCH','LEARNING_RATE','BATCH_SIZE','NUM_NETS','MODEL_ITERS','SAVE_ITERS',
                      'NUM_RAND_AVE','NUM_RAND_THRSH', 'NETWORKS'], [int(entries[0].get()), float(entries[1].get()),
                      int(entries[2].get()), int(entries[3].get()), int(entries[4].get()),
                      int(entries[5].get()), int(entries[6].get()), float(entries[7].get()), int(entries[8].get())]
        settings_dict = dict(zip(keys, value))
        program.set_params(settings_dict)
        self.dict = settings_dict
        settings.destroy()

    def start(self):
        self.cancel()
        self.destroy()
        self.original_frame.show()

    def train(self):
        print('Commence training')
        self.cancel()
        program.train_mrgan()

    def generate(self):
        print('Generating Image')
        self.cancel()
        if self.model_dir != '' or self.model_dir is not None:
            program.generate()
            self.set_image()
        else:
            self.text_box.insert(tk.INSERT, "\n\n!!! INVALID MODEL !!!\n\n")

    def open_folder(self, entry):
        directory = fl.askdirectory();
        entry.insert(0, directory)

    def apply_dir(self, directories, entries):
        self.set_dir = str(entries[0].get())
        self.model_dir = str(entries[1].get())
        self.out_dir = str(entries[2].get())
        program.set_folders(self.model_dir, self.set_dir, self.out_dir)
        self.set_image()
        self.load_config()
        directories.destroy()

    def open_dir(self):
        directories = tk.Tk()
        width, height = 480, 210
        screenwidth = directories.winfo_screenwidth()
        screenheight = directories.winfo_screenheight()
        directories.title("MrGAN Directories")
        directories.geometry('%dx%d+%d+%d' % (width, height, ((screenwidth / 2) - (width / 2)), ((screenheight / 2) - (height / 2))))
        directories.resizable(width=False, height=False)

        directories_frame = tk.Frame(directories, bg="plum")
        directories_frame.pack(fill=tk.BOTH)

        label_sets = tk.Label(directories_frame, font=20, text='Training Sets Folder:', bg="plum")
        entry_set_dir = tk.Entry(directories_frame, font=20, width=40)
        entry_set_dir.insert(0, '{}'.format(self.set_dir))
        btn_set_dir = tk.Button(directories_frame, font=6, text='Open', command=lambda: self.open_folder(entry_set_dir))
        label_model = tk.Label(directories_frame, font=20,  text='Model Folder:', bg="plum")
        entry_model_dir = tk.Entry(directories_frame, font=20, width=40)
        entry_model_dir.insert(0, '{}'.format(self.model_dir))
        btn_model_dir = tk.Button(directories_frame, font=6,  text='Open', command=lambda: self.open_folder(entry_model_dir))
        label_out = tk.Label(directories_frame, font=20,  text='Output Folder:', bg="plum")
        entry_out_dir = tk.Entry(directories_frame, font=20,  width=40)
        entry_out_dir.insert(0, '{}'.format(self.out_dir))
        btn_out_dir = tk.Button(directories_frame, font=6,  text='Open', command=lambda: self.open_folder(entry_out_dir))

        entries = [entry_set_dir, entry_model_dir, entry_out_dir]

        btn_dir_apply = tk.Button(directories_frame, width=8, font=6,  text='Apply', command=lambda: self.apply_dir(directories, entries))
        btn_dir_cancel = tk.Button(directories_frame, width=8, font=6,  text='Cancel', command=lambda: directories.destroy())

        label_sets.grid(row=0, column=0, columnspan=2, padx=20, sticky=tk.SW)
        entry_set_dir.grid(row=1, column=0, columnspan=2, padx=20, sticky=tk.NW)
        btn_set_dir.grid(row=1, column=3, padx=2)

        label_model.grid(row=2, column=0, columnspan=2, padx=20, sticky=tk.SW)
        entry_model_dir.grid(row=3, column=0, columnspan=2, padx=20, sticky=tk.NW)
        btn_model_dir.grid(row=3, column=3, padx=2)

        label_out.grid(row=4, column=0, columnspan=2, padx=20, sticky=tk.SW)
        entry_out_dir.grid(row=5, column=0, columnspan=2, padx=20, sticky=tk.NW)
        btn_out_dir.grid(row=5, column=3, padx=2)

        btn_dir_apply.grid(row=6, column=0, padx=3, pady=5)
        btn_dir_cancel.grid(row=6, column=1, padx=3, pady=5)

    def load_config(self):
        set_name = self.model_dir.split('/')[-1]
        config = self.model_dir+'/config.txt'
        if os.path.exists(config):
            params = {}
            config_txt = open(config, "r")
            for line in config_txt:
                parse = line.split('=')
                params[parse[0]] = parse[1].replace('\n', '')
            program.set_params(params)
            self.dict=params

    def cancel(self):
        if self._job is not None:
            root.after_cancel(self._job)
            self._job = None

    def update(self, ind, max_num):
        if ind+1 == max_num:
            ind = -1
        frame = self.frames[ind]
        ind += 1
        self.image_out_lbl.configure(image=frame)
        self._job = self.master.after(1500, self.update, ind, max_num)

    def set_image(self):
        self.cancel()
        if self.model_dir is not None and self.model_dir != '' :
            set_name = self.model_dir.split('/')[-1]
            temp_path = self.out_dir+'/training_image_output/'+set_name
            if os.path.exists(temp_path):
                files = os.listdir(temp_path)
                os.chdir(temp_path)
                self.frames = [ImageTk.PhotoImage(Image.open(i).resize((500, 350))) for i in files]
                self._job = self.master.after(0, self.update, 0, len(files)-1)
                os.chdir(self.cwd)
        else:
            photo = ImageTk.PhotoImage(Image.open("./source/placeholder.png").resize((400, 350)))
            self.model_dir = ''
            self.set_dir = ''

        # self.image_out_lbl.configure(image = photo)
        # self.image_out_lbl.image = photo

    def __init__(self, original):
        self.dict = {}
        self.dict['EPOCH'] = 500
        self.dict['LEARNING_RATE'] = 0.0002
        self.dict['BATCH_SIZE'] = 64
        self.dict['NUM_NETS'] = 3
        self.dict['MODEL_ITERS'] = 5
        self.dict['SAVE_ITERS'] = 5
        self.dict['NUM_RAND_AVE'] = 5
        self.dict['NUM_RAND_THRSH'] = 0.9
        self.dict['NETWORKS'] = 1

        self._job = None
        self.frames = None

        self.original_frame = original
        self.cwd = os.getcwd()
        self.model_dir = None
        self.set_dir = None
        self.out_dir = os.path.join(os.getcwd(),'output')
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
        self.image_menu = tk.PhotoImage(file='./source/main-menu.png')
        self.label_menu = tk.Label(self, image=self.image_menu)
        self.label_menu.place(x=0, y=0, width=800, height=500)

        # buttons
        self.image_open = tk.PhotoImage(file="./source/btn-main-open.png")
        self.button_open = tk.Button(self, image=self.image_open, borderwidth=0, command=lambda: self.open_dir())
        self.button_open.place(x=0, y=60, width=200, height=50)

        self.image_train = tk.PhotoImage(file="./source/btn-main-train.png")
        self.button_train = tk.Button(self, image=self.image_train, borderwidth=0, command=lambda: self.train())
        self.button_train.place(x=0, y=110, width=200, height=50)

        self.image_generate = tk.PhotoImage(file="./source/btn-main-generate.png")
        self.button_generate = tk.Button(self, image=self.image_generate, borderwidth=0, command=lambda: self.generate())
        self.button_generate.place(x=0, y=160, width=200, height=50)

        self.image_settings = tk.PhotoImage(file="./source/btn-main-settings.png")
        self.button_settings = tk.Button(self, image=self.image_settings, borderwidth=0, command=lambda: self.open_settings())
        self.button_settings.place(x=0, y=210, width=200, height=50)

        self.image_back = tk.PhotoImage(file="./source/btn-main-back.png")
        self.button_back = tk.Button(self, image=self.image_back, borderwidth=0, command=lambda: self.start())
        self.button_back.place(x=0, y=260, width=200, height=50)

        # window frame
        self.main_frame = tk.Frame(self, width=600, height=350)
        self.main_frame.place(x=200, y=50)

        # Output images
        self.image_out_frame = tk.Frame(self.main_frame)
        self.image_out_frame.place(x=0, y=0, width=600, height=350)

        self.image_out_lbl = tk.Label(self.image_out_frame)
        self.image_out_lbl.place(x=0, y=0, width=600, height=350)
        self.set_image()

        # logs
        self.text_frame_master = tk.Frame(self, bg="plum", width=600, height=100)
        self.text_frame_master.place(x=200, y=400)

        self.text_frame = tk.Frame(self.text_frame_master)
        self.text_frame.pack(fill=tk.BOTH, expand=True)

        self.text_box = tk.Text(self.text_frame, bg="plum", relief="sunken", width=83, height=6)
        self.text_box.config(font=("Arial", 10), undo=True, wrap='word')
        self.text_box.insert(tk.INSERT, "Welcome to MrGAN!\n")
        self.text_box.pack(side=tk.LEFT, fill=tk.BOTH)

        self.scroll = tk.Scrollbar(self.text_frame)
        self.scroll.config(command = self.text_box.yview)
        self.text_box.config(yscrollcommand = self.scroll.set)
        self.scroll.pack(side=tk.RIGHT, fill=tk.Y)

        #self.image_settings = tk.PhotoImage(file="./source/btn-settings.png")
        #self.button_settings = tk.Button(self.image_out_frame, image=self.image_settings, command=lambda: self.open_settings())
        #self.button_settings.place(x=0, y=0, width=120, height=26)

        program.init_objects(self.text_box, tk)

class startMenu(object):
    def about(self, master):
        self.master = master
        width = 800
        height = 500
        screenwidth = master.winfo_screenwidth()
        screenheight = master.winfo_screenheight()
        master.title("ABOUT")
        master.geometry('%dx%d+%d+%d' % (width, height, ((screenwidth / 2) - (width / 2)), ((screenheight / 2) - (height / 2))))
        master.resizable(width=False, height=False)

        self.image_entire = tk.PhotoImage(file="./source/about-menu.png")
        self.label_entire = tk.Label(master, image=self.image_entire)
        self.label_entire.place(x=0, y=0, width=800, height=500)

        self.image_home = tk.PhotoImage(file="./source/btn-about-home.png")
        self.button_home = tk.Button(master, image=self.image_home, borderwidth=0, command=lambda: self.__init__(master))
        self.button_home.place(x=0, y=0, width=75, height=50)

        self.image_start = tk.PhotoImage(file="./source/btn-about-start.png")
        self.button_start = tk.Button(root, image=self.image_start, borderwidth=0, command=lambda: self.open_main())
        self.button_start.place(x=75, y=0, width=75, height=50)

        self.image_tips = tk.PhotoImage(file="./source/btn-about-tips.png")
        self.button_tips = tk.Button(master, image=self.image_tips, borderwidth=0, command=lambda: self.tips(master))
        self.button_tips.place(x=150, y=0, width=75, height=50)

        self.image_exit = tk.PhotoImage(file="./source/btn-about-exit.png")
        self.button_exit = tk.Button(master, image=self.image_exit, borderwidth=0, command=lambda: self.about(master))
        self.button_exit.place(x=225, y=0, width=75, height=50)

        #with open("./source/about.txt", "r") as content_file:
        #    content = content_file.read()
        #msg.showinfo('About',content)

    def tips(self, master):
        self.master = master
        width = 800
        height = 500
        screenwidth = master.winfo_screenwidth()
        screenheight = master.winfo_screenheight()
        master.title("TIPS")
        master.geometry('%dx%d+%d+%d' % (width, height, ((screenwidth / 2) - (width / 2)), ((screenheight / 2) - (height / 2))))
        master.resizable(width=False, height=False)

        self.image_entire = tk.PhotoImage(file="./source/tips-menu.png")
        self.label_entire = tk.Label(master, image=self.image_entire)
        self.label_entire.place(x=0, y=0, width=800, height=500)

        self.image_home = tk.PhotoImage(file="./source/btn-tips-home.png")
        self.button_home = tk.Button(master, image=self.image_home, borderwidth=0, command=lambda: self.__init__(master))
        self.button_home.place(x=0, y=0, width=75, height=50)

        self.image_about = tk.PhotoImage(file="./source/btn-tips-about.png")
        self.button_about = tk.Button(root, image=self.image_about, borderwidth=0, command=lambda: self.about(master))
        self.button_about.place(x=75, y=0, width=75, height=50)

        self.image_start = tk.PhotoImage(file="./source/btn-tips-start.png")
        self.button_start = tk.Button(master, image=self.image_start, borderwidth=0, command=lambda: self.open_main())
        self.button_start.place(x=150, y=0, width=75, height=50)

        self.image_tips = tk.PhotoImage(file="./source/btn-tips-exit.png")
        self.button_tips = tk.Button(master, image=self.image_tips, borderwidth=0, command=lambda: self.tips(master))
        self.button_tips.place(x=225, y=0, width=75, height=50)

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
        self.image_entire =  tk.PhotoImage(file="./source/start-menu.png")
        self.label_entire =  tk.Label(master, image=self.image_entire)
        self.label_entire.place(x=0, y=0, width=800, height=500)

        #buttons
        self.image_about =  tk.PhotoImage(file="./source/btn-about.png")
        self.button_about =  tk.Button(master, image=self.image_about, borderwidth=0, command=lambda:self.about(master))
        self.button_about.place(x=300, y=0, width=250, height=250)

        self.image_tips =  tk.PhotoImage(file="./source/btn-tips.png")
        self.button_tips =  tk.Button(master, image=self.image_tips, borderwidth=0, command=lambda:self.tips(master))
        self.button_tips.place(x=550, y=0, width=250, height=250)

        self.image_start = tk.PhotoImage(file="./source/btn-start.png")
        self.button_start = tk.Button(root, image=self.image_start, borderwidth=0, command=lambda: self.open_main())
        self.button_start.place(x=300, y=250, width=500, height=250)

if __name__ == "__main__":
    root = tk.Tk()
    start_window = startMenu(root)
    root.mainloop()
