import tkinter as tk
from tkinter import ttk, scrolledtext
from data_function import upload_file, execute_model, execute_visualize, selected_vs_type

root = tk.Tk()
root.title('App version 1.3.6')


left_frame = tk.LabelFrame(root, text='Choose File')
left_frame.grid(row=0, column=0, padx=10, pady=10)

right_frame = tk.LabelFrame(root)
right_frame.grid(row=0, column=1, padx=10, pady=10)

tabControl = ttk.Notebook(right_frame)
dataTab = ttk.Frame(tabControl)
modelTab = ttk.Frame(tabControl)
visualizeTab = ttk.Frame(tabControl)
transformTab = ttk.Frame(tabControl)

tabControl.add(dataTab, text='Data')
tabControl.add(modelTab, text='Model')
tabControl.add(visualizeTab, text='Visualize')
tabControl.add(transformTab, text='Transform')
tabControl.grid(row=0, column=0, columnspan=2)

# Data tab
my_font1 = ('times', 12, 'bold')
path_label = tk.Label(left_frame, text='Read File & create DataFrame',
                      width=30, font=my_font1)
path_label.grid(row=1, column=1)
browse_btn = tk.Button(left_frame, text='Browse File',
                       width=20, command=lambda: upload_file())
browse_btn.grid(row=2, column=1, pady=5)

count_label = tk.Label(dataTab, width=40, text='',
                       bg='lightyellow')
count_label.grid(row=3, column=1, padx=5)
trv = ttk.Treeview(dataTab, selectmode='browse', height=10, show='headings')
trv.grid(row=5, column=1, columnspan=3, padx=10, pady=20)

# Model tab
target_label = tk.Label(modelTab, text="Select Target Variable")
target_label.grid(row=0, column=0, padx=5, sticky=tk.W)

target_combobox = ttk.Combobox(modelTab)
target_combobox.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)

input_label = tk.Label(modelTab, text="Select Input Variables")
input_label.grid(row=2, column=0, padx=5, sticky=tk.W)

input_label_cb = tk.Label(modelTab)
input_label_cb.grid(row=3, column=0, padx=5, sticky=tk.W)

model_label = tk.Label(modelTab, text="Choose Model")
model_label.grid(row=0, column=3, padx=50, pady=10, sticky=tk.W)

model_combobox = ttk.Combobox(
    modelTab, values=["Logistic Regression", "KNN", "Linear Regression", "Random Forest"]
)
model_combobox.grid(row=1, column=3, padx=50, sticky=tk.W)

execution_button = tk.Button(modelTab, text="Execution", command=execute_model)
execution_button.grid(row=2, column=3, padx=50, pady=10, sticky=tk.W)

# Visualize tab
visualize_title = tk.Label(visualizeTab, text="Visualize Type")
visualize_title.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

vs_type_combobox = ttk.Combobox(visualizeTab, values= ["1 Quantitative", "1 Categorical", "2 Categorical", 
                                                       "1 Categorical - 1 Quantitative", "2 Quantitative"], width=40)
vs_type_combobox.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
vs_type_combobox.bind("<<ComboboxSelected>>", selected_vs_type)

column1_label = tk.Label(visualizeTab, text="Column 1")
column1_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)

column2_label = tk.Label(visualizeTab, text="Column 2")
column2_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)

column1_combobox = ttk.Combobox(visualizeTab)
column1_combobox.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

column2_combobox = ttk.Combobox(visualizeTab)
column2_combobox.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)

vs_graph_type_label = tk.Label(visualizeTab, text="Graph Type")
vs_graph_type_label.grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)

vs_graph_type_combobox = ttk.Combobox(visualizeTab)
vs_graph_type_combobox.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)

excute_btn = tk.Button(visualizeTab, text="Execute", command=execute_visualize)
excute_btn.grid(row=4, column=1, padx=5, pady=5, sticky=tk.W)

root.mainloop()  # Keep the window open