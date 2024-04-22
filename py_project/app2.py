import pandas as pd
import tkinter as tk
from tkinter import *
from tkinter import filedialog, messagebox, ttk
import os
import traceback
import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from keras.utils import to_categorical
from keras_preprocessing.image import load_img
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input

from tqdm.notebook import tqdm

root = tk.Tk()
root.title('App version 1.3.6')

my_ref={} # to store references to checkboxes 
i=1
selected_checkboxes = [] # To store the checkbuttons which are checked

# Data functions
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
    target_combobox["values"] = tree_list
    my_columns()

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

    vs = ttk.Scrollbar(tab1, orient="vertical", command=trv.yview)
    trv.configure(yscrollcommand=vs.set)  # connect to Treeview
    vs.grid(row=5, column=4, sticky='ns')  # Place on grid

# Visualize functions
def my_columns():
    global i, my_ref, selected_checkboxes
    i = 1  # to increase the column number
    my_ref = {}  # to store references to checkboxes
    selected_checkboxes = []  # Initialize the list of selected checkboxes
    input_label_cb.config(text=" ")  # Remove the previous checkboxes
    for column in tree_list:
        var = IntVar()
        cb = Checkbutton(tab2, text=column, variable=var)
        cb.grid(row=i + 3, column=0, padx=5, sticky=tk.W)
        my_ref[column] = var
        i += 1
        selected_checkboxes.append((column, var))  # Append checkbox and its variable to the list

def execute_model():
    global model_train  # Di chuyển câu lệnh global lên đầu hàm
    target_variable = target_combobox.get()
    input_variables = [column for column, var in selected_checkboxes if var.get() == 1]
    le = LabelEncoder()

    # if input variable is categorical convert to numerical
    for column in input_variables:
        if df[column].dtype == "object":
            df[column] = le.fit_transform(df[column])
    if df[target_variable].dtype == "object":
        df[target_variable] = le.fit_transform(df[target_variable])
    # convert to list
    input_variables = list(input_variables)
    # delete item in list empty
    input_variables = [x for x in input_variables if x != ""]

    print(target_variable)
    print(input_variables)
    X = df[input_variables]
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = model_combobox.get()

    print("Executing model...")
    if model == "Logistic Regression":
        model_train = LogisticRegression()
        print("Logistic Regression")
    elif model == "KNN":
        model_train = KNeighborsClassifier()
        print("KNN")
    elif model == "Linear Regression":
        model_train = LinearRegression()
        print("Linear Regression")
    model_train.fit(X_train, y_train)
    y_pred = model_train.predict(X_test)

    if isinstance(model_train, LogisticRegression) or isinstance(
        model_train, KNeighborsClassifier
    ):
        try:
            accuracy = accuracy_score(y_test, y_pred)
            messagebox.showinfo("Model Result", f"Accuracy: {accuracy}")
        except:
            print("error")

    elif isinstance(model_train, LinearRegression):
        r2 = str(r2_score(y_test, y_pred))
        messagebox.showinfo("Model Result", f"Mean Squared Error: {r2}")

def create_cnn_df(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir,label)):
            image_paths.append(os.path.join(dir,label,imagename))
            labels.append(label)
        print(label, "completed")
    return image_paths,labels

def choose_train_dir():
    global train, TRAIN_DIR
    TRAIN_DIR = filedialog.askdirectory()
    train_dir_label.config(text=TRAIN_DIR)
    train = pd.DataFrame()
    train['image'], train['label'] = create_cnn_df(TRAIN_DIR)

def choose_test_dir():
    global test, TEST_DIR
    TEST_DIR = filedialog.askdirectory()
    test_dir_label.config(text=TEST_DIR)
    test = pd.DataFrame()
    test['image'], test['label'] = create_cnn_df(TEST_DIR)
    
def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, color_mode='grayscale', target_size=(48,48))
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features),48,48,1)
    return features

def train_model():
    train_features = extract_features(train['image'])
    test_features = extract_features(test['image'])
    
    x_train = train_features/255.0
    x_test = test_features/255.0
    
    le = LabelEncoder()
    le.fit(train['label'])
    
    y_train = le.transform(train['label'])
    y_test = le.transform(test['label'])
    
    y_train = to_categorical(y_train, num_classes=7)
    y_test = to_categorical(y_test, num_classes=7)
    
    model = Sequential()
    model.add(Input(shape=(48,48,1)))

    model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(7, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
    
# GUI
left_frame = tk.LabelFrame(root, text='Choose File')
left_frame.grid(row=0, column=0, padx=10, pady=10)

right_frame = tk.LabelFrame(root)
right_frame.grid(row=0, column=1, padx=10, pady=10)

tabControl = ttk.Notebook(right_frame)
tab1 = ttk.Frame(tabControl)
tab2 = ttk.Frame(tabControl)
tab3 = ttk.Frame(tabControl)

tabControl.add(tab1, text='Data')
tabControl.add(tab2, text='Visualize')
tabControl.add(tab3, text='CNN model for classification')
tabControl.grid(row=0, column=0, columnspan=2)

# Data tab
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
search_entry = tk.Entry(tab1, width=35, font=18)  # added one Entry box
search_entry.grid(row=4, column=1, padx=1)

# Visualize tab
target_label = tk.Label(tab2, text="Select Target Variable")
target_label.grid(row=0, column=0, padx=5, sticky=tk.W)

target_combobox = ttk.Combobox(tab2)
target_combobox.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)

input_label = tk.Label(tab2, text="Select Input Variables")
input_label.grid(row=2, column=0, padx=5, sticky=tk.W)

input_label_cb = tk.Label(tab2)
input_label_cb.grid(row=3, column=0, padx=5, sticky=tk.W)

model_label = tk.Label(tab2, text="Choose Model")
model_label.grid(row=0, column=3, padx=50, pady=10, sticky=tk.W)

model_combobox = ttk.Combobox(
    tab2, values=["Logistic Regression", "KNN", "Linear Regression"]
)
model_combobox.grid(row=1, column=3, padx=50, sticky=tk.W)

execution_button = tk.Button(tab2, text="Execution", command=execute_model)
execution_button.grid(row=2, column=3, padx=50, pady=10, sticky=tk.W)

# CNN tab

train_dir_label = tk.Label(tab3, text="Train Directory")
train_dir_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

train_dir_btn = tk.Button(tab3, text="Browse", command=choose_train_dir)
train_dir_btn.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

test_dir_label = tk.Label(tab3, text="Test Directory")
test_dir_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)

test_dir_btn = tk.Button(tab3, text="Browse", command=choose_test_dir)
test_dir_btn.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

train_model_btn = tk.Button(tab3, text="Train", width=20, command=train_model)
train_model_btn.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)

root.mainloop()  # Keep the window open
