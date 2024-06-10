import os
from tkinter import Checkbutton, IntVar, filedialog

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def upload_file():
    from gui import trv, count_label, path_label, target_combobox  # Di chuyển import vào hàm upload_to_GUI
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
    
    target_combobox["values"] = tree_list
    my_columns()

# Model functions
def my_columns():
    from gui import input_label_cb, modelTab
    global i, my_ref, selected_checkboxes
    i = 1  # to increase the column number
    my_ref = {}  # to store references to checkboxes
    selected_checkboxes = []  # Initialize the list of selected checkboxes
    input_label_cb.config(text=" ")  # Remove the previous checkboxes
    for column in tree_list:
        var = IntVar()
        cb = Checkbutton(modelTab, text=column, variable=var)
        cb.grid(row=i + 3, column=0, padx=5)
        my_ref[column] = var
        i += 1
        # Append checkbox and its variable to the list
        selected_checkboxes.append((column, var))

def execute_model():
    from gui import model_combobox, target_combobox
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
        X, y, test_size=0.33, random_state=42
    )
    
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    
    #future scaling
    st_x= StandardScaler()    
    X_train= st_x.fit_transform(X_train)    
    X_test= st_x.transform(X_test)  

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
    elif model == "Decision Tree":
        model_train = DecisionTreeClassifier(criterion="entropy")
        print("Decision Tree")
    elif model == "Random Forest":
        model_train = RandomForestClassifier(n_estimators= 10, criterion="entropy")  
        print("Random Forest")
    
    model_train.fit(X_train, y_train)
    y_pred = model_train.predict(X_test)
    print("y_pred:", y_pred)

    if isinstance(model_train, LogisticRegression) or isinstance(model_train, KNeighborsClassifier):
        try:
            accuracy = accuracy_score(y_test, y_pred)
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.text(0.5, 0.95, f"Accuracy: {accuracy}", ha='center', va='top', transform=plt.gca().transAxes, fontsize=10)
            plt.show()
        except Exception as e:
            print("Error:", e)
    elif isinstance(model_train, LinearRegression):
        # Calculate R-squared
        r2 = r2_score(y_test, y_pred)
        # Scatter plot
        plt.scatter(y_test, y_pred)
        plt.plot(y_test, y_test, color='red', linewidth=2)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Scatter plot")
        plt.text(0.5, 0.95, f"R-squared: {r2}", ha='center', va='top', transform=plt.gca().transAxes, fontsize=10)
        plt.show()
    elif isinstance(model_train, DecisionTreeClassifier) or isinstance(model_train, RandomForestClassifier):
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        X_set, y_set = X_test, y_test  
        x1, x2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step  =0.01), 
                            np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(x1, x2, model_train.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
                     alpha = 0.75, cmap = ListedColormap(('purple', 'green')))
        plt.xlim(x1.min(), x1.max())
        plt.ylim(x2.min(), x2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],  
                        c = ListedColormap(('purple', 'green'))(i), label = j)
        plt.title('Random Forest Classification')
        plt.xlabel('Independent ')
        plt.ylabel('Dependent')
        plt.legend()
        plt.text(0.5, 0.95, f"Accuracy: {accuracy}", ha='center', va='top', transform=plt.gca().transAxes, fontsize=10)
        plt.show()

# Visualize functions
def selected_vs_type(event):
    from gui import vs_type_combobox, column1_label, column1_combobox, column2_label, column2_combobox, vs_graph_type_combobox
    quantitative_columns = [column for column in tree_list if df[column].dtype in ['int64', 'float64']]
    categorical_columns = [column for column in tree_list if df[column].dtype == 'object']
    graph_type_1 = ["Histogram", "Box Plot"]
    graph_type_2 = ["Bar Chart", "Pie Chart"]
    graph_type_3 = ["Stacked Bar Chart", "Heatmap"]
    graph_type_4 = ["Bar Chart", "Violin Plot"]
    graph_type_5 = ["Line Plot", "Scatter Plot"]
      
    selected = vs_type_combobox.get()
    if selected == "1 Quantitative":
        column1_label.config(text="Quantitative")
        column1_combobox["values"] = quantitative_columns
        
        column2_combobox["values"] = []
        column2_label.grid_remove()
        column2_combobox.grid_remove()
        
        vs_graph_type_combobox["values"] = graph_type_1
    elif selected == "1 Categorical":
        column1_label.config(text="Categorical")
        column1_combobox["values"] = categorical_columns
        
        column2_combobox["values"] = []
        column2_label.config(text="")
        column2_combobox.grid_remove()
        column2_combobox.grid_remove()
        
        vs_graph_type_combobox["values"] = graph_type_2
    elif selected == "2 Categorical":
        column1_label.config(text="Categorical 1")
        column2_label.config(text="Categorical 2")
        
        column1_combobox["values"] = categorical_columns
        column2_combobox["values"] = categorical_columns
        
        column2_label.grid(row=2, column=0, padx=5, pady=5)
        column2_combobox.grid(row=2, column=1, padx=5, pady=5)
        
        vs_graph_type_combobox["values"] = graph_type_3
    elif selected == "1 Categorical - 1 Quantitative":
        column1_label.config(text="Categorical")
        column2_label.config(text="Quantitative")
        
        column1_combobox["values"] = categorical_columns
        column2_combobox["values"] = quantitative_columns
        
        column2_label.grid(row=2, column=0, padx=5, pady=5)
        column2_combobox.grid(row=2, column=1, padx=5, pady=5)
        
        vs_graph_type_combobox["values"] = graph_type_4
    elif selected == "2 Quantitative":
        column1_label.config(text="Quantitative 1")
        column2_label.config(text="Quantitative 2")
        
        column1_combobox["values"] = quantitative_columns
        column2_combobox["values"] = quantitative_columns
        
        column2_label.grid(row=2, column=0, padx=5, pady=5)
        column2_combobox.grid(row=2, column=1, padx=5, pady=5)
        
        vs_graph_type_combobox["values"] = graph_type_5
        
def bar_chart_1_column(column):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=column)
    plt.title(f"Bar chart for {column}")
    plt.show()
    
def histogram_1_column(column):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=column, kde=True)
    plt.title(f"Histogram for {column}")
    plt.show()
    
def pie_chart_1_column(column):
    plt.figure(figsize=(10, 6))
    df[column].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title(f"Pie chart for {column}")
    plt.show()
    
def box_plot_1_column(column):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, y=column)
    plt.title(f"Box plot for {column}")
    plt.show()
    
def stacked_bar_chart_2_columns(column1, column2):
    plt.figure(figsize=(10, 6))
    df.groupby(column1)[column2].value_counts().unstack().plot(kind='bar', stacked=True)
    plt.title(f"Stacked bar chart for {column1} and {column2}")
    plt.show()
    
def heatmap_2_columns(column1, column2):
    plt.figure(figsize=(10, 6))
    sns.heatmap(pd.crosstab(df[column1], df[column2]), annot=True, fmt='d')
    plt.title(f"Heatmap for {column1} and {column2}")
    plt.show()

def box_plot_2_columns(column1, column2):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=column1, y=column2)
    plt.title(f"Box plot for {column1} and {column2}")
    plt.show()
    
def violin_plot_2_columns(column1, column2):
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x=column1, y=column2)
    plt.title(f"Violin plot for {column1} and {column2}")
    plt.show()
    
def line_plot_2_columns(column1, column2):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x=column1, y=column2)
    plt.title(f"Line plot for {column1} and {column2}")
    plt.show()    
    
def bar_chart_2_columns(column1, column2):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=column1, hue=column2)
    plt.title(f"Bar chart for {column1} and {column2}")
    plt.show()
    
def scatter_plot_2_columns(column1, column2):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=column1, y=column2)
    plt.title(f"Scatter plot for {column1} and {column2}")
    plt.show()
    
def box_plot_2_columns(column1, column2):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=column1, y=column2)
    plt.title(f"Box plot for {column1} and {column2}")
    plt.show()
    
def execute_visualize():
    from gui import vs_type_combobox, column1_combobox, column2_combobox, vs_graph_type_combobox
    vs_type = vs_type_combobox.get()
    column1 = column1_combobox.get()
    column2 = column2_combobox.get()
    graph_type = vs_graph_type_combobox.get()
    
    if vs_type == "1 Quantitative":
        if graph_type == "Histogram":
            histogram_1_column(column1)
        elif graph_type == "Box Plot":
            box_plot_1_column(column1)
    elif vs_type == "1 Categorical":
        if graph_type == "Bar Chart":
            bar_chart_1_column(column1)
        elif graph_type == "Pie Chart":
            pie_chart_1_column(column1)
    elif vs_type == "2 Categorical":
        if graph_type == "Stacked Bar Chart":
            stacked_bar_chart_2_columns(column1, column2)
        elif graph_type == "Heatmap":
            heatmap_2_columns(column1, column2)
    elif vs_type == "1 Categorical - 1 Quantitative":
        if graph_type == "Bar Chart":
            bar_chart_2_columns(column1, column2)
        elif graph_type == "Violin Plot":
            violin_plot_2_columns(column1, column2)
    else:
        if graph_type == "Line Plot":
            line_plot_2_columns(column1, column2)
        elif graph_type == "Scatter Plot":
            scatter_plot_2_columns(column1, column2)
    