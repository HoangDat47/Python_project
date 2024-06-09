import os
from tkinter import filedialog

import pandas as pd

def upload_file():
    from gui import trv, count_label, path_label  # Di chuyển import vào hàm upload_to_GUI
    global file_name, str1, df, tree_list
    f_types = [('CSV files', "*.csv"), ('All', "*.*")]
    file = filedialog.askopenfilename(filetypes=f_types)
    file_name = os.path.basename(file)
    df = pd.read_csv(file)
    tree_list = list(df)  # List of column names as header
    str1 = "Rows:" + str(df.shape[0]) + " , Columns:" + str(df.shape[1])
    
    path_label.config(text=file_name)
    count_label.config(text=str1)
    trv.config(columns=tree_list)
    
    # Xóa tất cả các cột cũ trong treeview (nếu có)
    for col in trv["columns"]:
        trv.column(col, width=0)
        trv.heading(col, text="")
    
    # Xóa tất cả các dòng cũ trong treeview (nếu có)
    trv.delete(*trv.get_children())
    
    # Thiết lập các cột của treeview dựa trên tree_list
    for col in tree_list:
        trv.heading(col, text=col)
        trv.column(col, width=100)  # Thiết lập độ rộng của cột
    
    # Thêm dữ liệu từ DataFrame vào treeview
    for i, row in df.iterrows():
        trv.insert("", "end", values=list(row))
    