from tkinter.filedialog import asksaveasfile

def save(): 
    files = [('JPEG File', '*.jpg'), 
             ('PNG File', '*.png')] 
    file = asksaveasfile(filetypes = files, defaultextension = files) 
    return file