from tkinter import *
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog
import h5py
import os

img_arr = np.zeros((28,28,3), dtype=np.float32)
img_arr[0,0,0] = 1
img_arr = img_arr * 255.0
img_arr = img_arr.astype(np.uint8)

# img = Image.fromarray(img_arr, mode='RGB')
# img = img.resize((128, 128), Image.ANTIALIAS)
# pi = ImageTk.PhotoImage(image=img)

class App:
    def __init__(self, master):
        self.frame = Frame(master)
        self.frame.pack()

        self.init_file_chooser()
        self.init_data_viewer()

        self.x_train = None
        self.y_train = None
        

    def init_file_chooser(self):
        self.file_chooser_frame = Frame(self.frame)
        f = self.file_chooser_frame
        f.pack(side=TOP)

        Label(f, text="Data Set File:").grid(row=0, column=0)     
        tb = Entry(f)
        tb.grid(row=0, column=1)
        tb.bind('<Double-Button-1>', self.open_file_chooser_dialog)
        self.file_chooser_tb = tb

        btn = Button(f, text="Load", command=self.load_dataset)
        btn.grid(row=0, column=2)
        self.file_chooser_btn = btn

        lbl = Label(f, text='')
        lbl.grid(row=0, column=3)
        self.file_chooser_lbl = lbl
        
    def open_file_chooser_dialog(self, e):
        tb = self.file_chooser_tb
        fname = filedialog.askopenfilename()
        
        if fname == '':
            return
        
        tb.delete(0,'end')
        tb.insert(0, fname)

    def load_dataset(self):
        fname = self.file_chooser_tb.get()
        self.file_chooser_lbl.config(text='Loading...')
        self.file_chooser_lbl.update()

        if os.path.isfile(fname) == False:
            self.file_chooser_lbl.config(text='Not found!')
            return

        with h5py.File(fname) as f:
            self.x_train = np.array(f['images'])
            self.y_train = np.array(f['labels'])

        print('x_train shape: {}'.format(self.x_train.shape))
        print('y_train shape: {}'.format(self.y_train.shape))

        self.num_images = self.x_train.shape[0]
        self.data_nav_sl.config(to=self.num_images)
        self.patch_size = self.x_train.shape[1]
        self.img_idx = 0
        self.update_img()
        self.file_chooser_lbl.config(text='Done!')
        self.frame.update()


    def create_tkimage(self, img_arr):
        if np.max(img_arr) <= 1.0:
            img_arr = img_arr * 255.0

        img_arr = img_arr.astype(np.uint8)

        if img_arr.shape[2] == 3:
            img = Image.fromarray(img_arr, mode='RGB')
        else:
            img_arr = img_arr.reshape((img_arr.shape[0], img_arr.shape[1]))
            img = Image.fromarray(img_arr, mode='L')

        img = img.resize((512, 512), Image.NEAREST)
            
        return ImageTk.PhotoImage(image=img)

    def update_img(self):
        img_arr_x = self.x_train[self.img_idx]
        img_arr_y = self.y_train[self.img_idx]

        self.img_x = self.create_tkimage(img_arr_x)
        self.img_y = self.create_tkimage(img_arr_y)

        self.img_viewer_x_lbl.config(image=self.img_x)
        self.img_viewer_y_lbl.config(image=self.img_y)
        self.data_viewer_frame.update()

    def init_data_viewer(self):
        self.data_viewer_frame = Frame(self.frame)
        self.data_viewer_frame.pack(side=TOP)
        self.init_img_viewer()
        self.init_data_nav()

    def init_img_viewer(self):
        self.img_viewer_frame = Frame(self.data_viewer_frame)
        self.img_viewer_frame.pack(side=TOP)
        f = self.img_viewer_frame

        self.dummy_img = self.create_tkimage(np.zeros((512, 512, 3)))

        x_lbl = Label(f, image=self.dummy_img)
        x_lbl.grid(row=0, column=0)
        self.img_viewer_x_lbl = x_lbl

        y_lbl = Label(f, image=self.dummy_img)
        y_lbl.grid(row=0, column=1)
        self.img_viewer_y_lbl = y_lbl

    def init_data_nav(self):
        f = self.data_viewer_frame

        sl = Scale(f, from_=0, orient=HORIZONTAL, length=1024, command=self.slider_changed)
        sl.pack(side=BOTTOM)

        self.data_nav_sl = sl

    def slider_changed(self, e):
        self.img_idx = self.data_nav_sl.get()

        if self.x_train != None:
            self.update_img()

root = Tk()
app = App(root)
root.mainloop()