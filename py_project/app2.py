import pandas as pd
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import os

root = tk.Tk()
root.title('App version 1.3.2')

left_frame = tk.LabelFrame(root, text='Choose File')
left_frame.grid(row=0, column=0, padx=10, pady=10)

right_frame = tk.LabelFrame(root, text='Model')
right_frame.grid(row=0, column=1, padx=10, pady=10)

tabControl = ttk.Notebook(right_frame)
tab1 = ttk.Frame(tabControl)
tab2 = ttk.Frame(tabControl)

tabControl.add(tab1, text='Data')
tabControl.add(tab2, text='Visualize')
tabControl.grid(row=0, column=0, columnspan=2)

my_font1 = ('times', 12, 'bold')
path_label = tk.Label(left_frame, text='Read File & create DataFrame',
                      width=30, font=my_font1)
path_label.grid(row=1, column=1)
browse_btn = tk.Button(left_frame, text='Browse File',
                       width=20, command=lambda: upload_file())
browse_btn.grid(row=2, column=1, pady=5)
count_label = tk.Label(tab1, width=40, text='',
                       bg='lightyellow')
count_label.grid(row=3, column=1, padx=5)
search_entry = tk.Entry(tab1, width=35, bg="yellow", font=18)  # added one Entry box
search_entry.grid(row=4, column=1, padx=1)

def upload_file():
    global df, tree_list
    f_types = [('CSV files', "*.csv"), ('All', "*.*")]
    file = filedialog.askopenfilename(filetypes=f_types)
    file_name = os.path.basename(file)
    path_label.config(text=file_name)
    df = pd.read_csv(file)
    tree_list = list(df)  # List of column names as header
    str1 = "Rows:" + str(df.shape[0]) + " , Columns:" + str(df.shape[1])
    count_label.config(text=str1)
    trv_refresh()  # show Treeview
    search_entry.bind('<KeyRelease>', lambda event: my_search())

def my_search():
    query = search_entry.get().strip().lower()  # Lấy giá trị từ Entry và chuyển về chữ thường
    if query:  # Kiểm tra xem giá trị được nhập vào hay không
        keywords = query.split()  # Tách các từ khóa
        # Tạo điều kiện tìm kiếm
        condition = df.apply(lambda row: all(keyword.lower() in str(row).lower() for keyword in keywords), axis=1)
        df2 = df[condition]  # Lọc các hàng thỏa mãn điều kiện tìm kiếm
        r_set = df2.to_numpy().tolist()  # Tạo danh sách các hàng kết quả

        # Cập nhật Treeview để hiển thị kết quả tìm kiếm
        trv_refresh(r_set)
    else:
        # Nếu không có giá trị tìm kiếm được nhập vào, hiển thị toàn bộ dữ liệu
        trv_refresh()

def trv_refresh(r_set=None):  # Refresh the Treeview to reflect changes
    global df, trv, tree_list
    if r_set is None:
        r_set = df.to_numpy().tolist()  # create list of list using rows

    if hasattr(root, 'trv'):
        root.trv.destroy()

    trv = ttk.Treeview(tab1, selectmode='browse', height=10,
                       show='headings', columns=tree_list)
    trv.grid(row=5, column=1, columnspan=3, padx=10, pady=20)

    for i in tree_list:
        trv.column(i, width=90, anchor='c')
        trv.heading(i, text=str(i))

    for dt in r_set:
        v = [r for r in dt]
        trv.insert("", 'end', iid=v[0], values=v)

    vs = ttk.Scrollbar(root, orient="vertical", command=trv.yview)
    trv.configure(yscrollcommand=vs.set)  # connect to Treeview
    vs.grid(row=5, column=4, sticky='ns')  # Place on grid


root.mainloop()  # Keep the window open
