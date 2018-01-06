from tkinter import Frame, Canvas, Scrollbar, VERTICAL, HORIZONTAL, N, S, E, W, Tk, Menu, filedialog, Button, RIGHT, \
    Label, Listbox, X, Entry, LEFT, Radiobutton, IntVar, BOTTOM, StringVar, messagebox

import PIL
from PIL import ImageTk

from seam_carver import generate_mask, SeamCarver


class Gui(Frame):
    def __init__(self, master):
        Frame.__init__(self, master=None)
        self.sbarv = Scrollbar(self, orient=VERTICAL)
        self.sbarh = Scrollbar(self, orient=HORIZONTAL)
        self.x = self.y = 0
        self.canvas = Canvas(self, cursor="cross")

        # Parameters to capture mouse movement
        self.rect = None
        self.im = None
        self.img_path = None

        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None

        self.radio_value = IntVar()
        self.width_value = StringVar()
        self.height_value = StringVar()
        self.out_file_name = StringVar()

        self.canvas.grid(row=0, column=0, sticky=N + S + E + W)

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # Menu bar on top
        menu_bar = Menu(master)
        master.config(menu=menu_bar)

        file_menu = Menu(menu_bar)
        file_menu.add_command(label="Open", command=self.on_open)
        menu_bar.add_cascade(label="File", menu=file_menu)

        # Buttons and labels on the right
        list_box = Listbox(master)
        list_box.pack(side=RIGHT, anchor=N)

        self.image_width_label = Label(list_box, text="Width: -")
        self.image_width_label.pack(fill=X)

        self.image_height_label = Label(list_box, text="Height: -")
        self.image_height_label.pack(fill=X)

        remove_button = Radiobutton(list_box, text="Remove selection", variable=self.radio_value, value=1)
        remove_button.pack(fill=X)

        protect_button = Radiobutton(list_box, text="Protect selection", variable=self.radio_value, value=2)
        protect_button.pack(fill=X)

        width_label = Label(list_box, text="width:", justify=LEFT)
        width_label.pack(fill=X)

        width_input = Entry(list_box, textvariable=self.width_value)
        width_input.pack(fill=X)

        height_label = Label(list_box, text="height:", justify=LEFT)
        height_label.pack(fill=X)

        height_input = Entry(list_box, textvariable=self.height_value)
        height_input.pack(fill=X)

        out_file_label = Label(list_box, text="filename (optional):", justify=LEFT)
        out_file_label.pack(fill=X)

        out_file_input = Entry(list_box, textvariable=self.out_file_name)
        out_file_input.pack(fill=X)

        start_button = Button(list_box, text="Start", height=10, width=10,
                              command=self.start_process)
        start_button.pack(fill=X)

    def on_open(self):
        ftypes = [('Image files', '*.jpg *.png'), ('All files', '*')]
        dlg = filedialog.Open(self, filetypes=ftypes)
        file_path = dlg.show()

        if file_path:
            self.img_path = file_path
            self.im = PIL.Image.open(file_path)
            width, height = self.im.size

            # Defined for scrolling if image is very large
            if width > 1280 or height > 768:
                self.sbarv.config(command=self.canvas.yview)
                self.sbarh.config(command=self.canvas.xview)
                self.canvas.config(yscrollcommand=self.sbarv.set)
                self.canvas.config(xscrollcommand=self.sbarh.set)
                self.sbarv.grid(row=0, column=1, stick=N + S)
                self.sbarh.grid(row=1, column=0, sticky=E + W)

            self.canvas.config(scrollregion=(0, 0, width, height))
            self.canvas.config(width=min(width, 1280), height=min(height, 768))
            self.image_width_label.config(text='Width: ' + str(width) + "px")
            self.image_height_label.config(text='Height: ' + str(height) + "px")

            self.tk_im = ImageTk.PhotoImage(self.im)
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_im)

    def on_button_press(self, event):
        if self.im:
            # save mouse drag start position
            self.start_x = int(self.canvas.canvasx(event.x))
            self.start_y = int(self.canvas.canvasy(event.y))

            # create rectangle if not yet exist
            if not self.rect:
                self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, outline='red')

    def on_move_press(self, event):
        if self.im:
            cur_x = self.canvas.canvasx(event.x)
            cur_y = self.canvas.canvasy(event.y)

            w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
            if event.x > 0.9 * w:
                self.canvas.xview_scroll(1, 'units')
            elif event.x < 0.1 * w:
                self.canvas.xview_scroll(-1, 'units')
            if event.y > 0.9 * h:
                self.canvas.yview_scroll(1, 'units')
            elif event.y < 0.1 * h:
                self.canvas.yview_scroll(-1, 'units')

            # expand rectangle as you drag the mouse
            self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        if self.im:
            self.end_x = int(self.canvas.canvasx(event.x))
            self.end_y = int(self.canvas.canvasx(event.y))

    def start_process(self):
        width, height, option = self.width_value.get(), self.height_value.get(), self.radio_value.get()
        if is_number(width) and is_number(height) and option in {0, 1, 2}:
            width, height, option = int(width), int(height), int(option)
            mask = None
            if option == 1:  # Remove selection
                mask = generate_mask(self.start_x, self.start_y, self.end_x, self.end_y, self.im, False)
            elif option == 2:  # Protect selection
                mask = generate_mask(self.start_x, self.start_y, self.end_x, self.end_y, self.im, True)

            output_filename = self.out_file_name.get()
            carver = SeamCarver(self.img_path, int(height), int(width), mask)
            if output_filename == "":
                output_filename = "output.jpg"
            else:
                output_filename += ".jpg"
            carver.save_result(output_filename)
            messagebox.showinfo("Done", "Image resizing complete, file saved in " + output_filename)
        else:
            messagebox.showwarning("Oops", "Output dimensions must be integers!")


def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    root = Tk()
    app = Gui(root)
    app.pack()
    root.mainloop()
