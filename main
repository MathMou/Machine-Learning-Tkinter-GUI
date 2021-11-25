import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import Toplevel
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showinfo, showwarning, askquestion
from tkinter import OptionMenu
from tkinter import StringVar

import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import style
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
from psutil import cpu_percent
from psutil import virtual_memory
from datetime import datetime, timedelta

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import sklearn.cluster as cluster
import scipy.spatial.distance as sdist
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import seaborn as sn

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42 #used to help randomly select the data points
low_memory=False
LARGE_FONT= ("Verdana", 12)
style.use("ggplot")

f = Figure(figsize=(5,5), dpi=100)
a = f.add_subplot(111)

df = None
df_nan = None
df_r = None
check_transformer = None
df_output = None
ContNum = None
GB_Classifier_score = None


'''

┏━━━━•❍•°•❍•°•❍•━━━━┓
❍          FUNCTIONS        ❍
┗━━━━•❍•°•❍•°•❍•━━━━┛


'''

def Credits():
    showinfo("INFO", 
             '''
━━━━━━•❍•°•❍•°•❍•━━━━━━
Goodbye & have a nice day!

CREATED BY MathMou     
 ━━━━━•❍•°•❍•°•❍•━━━━━━
             ''')
    app.destroy()
    
def RunUC():
    showwarning("Warning", "Under construction")

def open_file():
    global df, df_nan, df_r, df_output, best_n_clusters
    
    df = None
    df_nan = None
    df_r = None
    check_transformer = None
    df_output = None
    ClusterNum = None
    C_Number = None
    name = askopenfilename()

    if name:
        df = pd.read_excel(name, sep=';')
        showinfo("INFO", "DataFrame created")

def ResetterDF():
    global df, df_nan, df_r, ClusterNum, check_transformer
    
    df = None
    df_nan = None
    df_r = None
    check_transformer = None
    ClusterNum = None
    showinfo("INFO", "Dataframe reset")        
        
def NaNifier():
    global df, df_nan
    
    df_nan = None
    
    if df is None:
        showwarning("Warning", "Read file first")
    else:
        df_nan = df
        df_nan.replace(np.nan, 0, inplace=True)
        showinfo("INFO", "Blank values replaced by zero values")
        
def data_remover():
    global df_nan, df_r
    
    if df is None:
        #print("Read file first")
        showwarning("Warning", "Read file first")
    elif df_nan is None:
        #print("Read file first")
        showwarning("Warning", "Replace blank values first")
    elif df_r is not None:
        showwarning("Warning", "Data already cleaned")
    else:
        for y in df_nan.columns:
            if df_nan[y].dtype != np.float64:
                df_r = df_nan.select_dtypes(include=['float64', 'int64'])
            elif df_nan[y].dtype != np.int64:
                df_r = df_nan.select_dtypes(include=['float64', 'int64'])
            else:
                df_r = df_nan
            showinfo("INFO", "Data cleaned")

def MM_data_transformer():
    
    global df, df_nan, df_r, check_transformer
    
    MM_scaler = MinMaxScaler()
    
    if df is None:
        showwarning("Warning", "Read file first")
    elif df_nan is None:
        showwarning("Warning", "Replace blank values first")
    elif df_r is None:
        showwarning("Warning", "Clean data first")
    elif check_transformer is not None:
        showwarning("Warning", "Data already transformed")
    else:
        check_transformer = 1337 # Check to see if data is transformed
        df_r = df_r.astype(float)
        df_r = pd.DataFrame(MM_scaler.fit_transform(df_r),columns = df_r.columns) #Scaling data
        df_r = df_r.round(2) #Rounding data
        print(df_r)
        showinfo("INFO", "Data scaled")

def S_data_transformer():
    
    global df, df_nan, df_r, check_transformer
    
    S_scaler = StandardScaler()
    
    if df is None:
        showwarning("Warning", "Read file first")
    elif df_nan is None:
        showwarning("Warning", "Replace blank values first")
    elif df_r is None:
        showwarning("Warning", "Clean data first")
    elif check_transformer is not None:
        showwarning("Warning", "Data already transformed")
    else:
        check_transformer = 1337 # Check to see if data is transformed
        df_r = df_r.astype(float)
        df_r = pd.DataFrame(S_scaler.fit_transform(df_r),columns = df_r.columns) #Scaling data
        df_r = df_r.round(2) #Rounding data
        print(df_r)
        showinfo("INFO", "Data scaled")

def AutoCleaner_MM():
    global df, df_nan, df_r, check_transformer
    
    MM_scaler = MinMaxScaler()
    
    df_nan = None
    df_r = None
    check_transformer = None
    
    if df is None:
        showwarning("Warning", "Read file first")
    else:        
        df_nan = df
        df_nan.replace(np.nan, 0, inplace=True)
        
        for y in df_nan.columns:
            if df_nan[y].dtype != np.float64:
                df_r = df_nan.select_dtypes(include=['float64', 'int64'])
            elif df_nan[y].dtype != np.int64:
                df_r = df_nan.select_dtypes(include=['float64', 'int64'])
            else:
                df_r = df_nan
                
        check_transformer = 1337 # Check to see if data is transformed
        df_r = df_r.astype(float)
        df_r = pd.DataFrame(MM_scaler.fit_transform(df_r),columns = df_r.columns) #Scaling data
        df_r = df_r.round(2) #Rounding data
        showinfo("INFO", "Data transformed with MinMax scaling")

def AutoCleaner_S():
    global df, df_nan, df_r, check_transformer
    
    S_scaler = StandardScaler()
    
    df_nan = None
    df_r = None
    check_transformer = None
    
    if df is None:
        showwarning("Warning", "Read file first")
    else:        
        df_nan = df
        df_nan.replace(np.nan, 0, inplace=True)
        
        for y in df_nan.columns:
            if df_nan[y].dtype != np.float64:
                df_r = df_nan.select_dtypes(include=['float64', 'int64'])
            elif df_nan[y].dtype != np.int64:
                df_r = df_nan.select_dtypes(include=['float64', 'int64'])
            else:
                df_r = df_nan
                
        check_transformer = 1337 # Check to see if data is transformed
        df_r = df_r.astype(float)
        df_r = pd.DataFrame(S_scaler.fit_transform(df_r),columns = df_r.columns) #Scaling data
        df_r = df_r.round(2) #Rounding data
        showinfo("INFO", "Data transformed with StandardScalar")
        
def RunSilHou():
    #Silhouette Score for further clustering
    global df, df_nan, df_r, df_output, ClusterNum
    
    ClusterNum = None
    
    if df is None:
        showwarning("Warning", "Read file first")
    elif df_nan is None:
        showwarning("Warning", "Replace blank values first")
    elif df_r is None:
        showwarning("WARNING", "Clean file first")
    elif df_output is not None:
        showwarning("WARNING", "Already computed KMeans")
    else:
        n_samples, n_features = df_r.shape
        procent_samples = n_samples
        #procent_samples = n_samples*0.10 #10% of the dataset for sample_size
        procent_samples = int(procent_samples)

        sil_score_max = -1 #this is the minimum possible score

        for n_clusters in range(2,10):
          model = KMeans(n_clusters = n_clusters, n_jobs = 8, init='k-means++', max_iter=100, n_init=1)
          labels = model.fit_predict(df_r)
          sil_score = silhouette_score(df_r, labels, sample_size = procent_samples, random_state = RANDOM_STATE)
          print("The average silhouette score for %i clusters is %0.10f" %(n_clusters, sil_score))
          if sil_score > sil_score_max:
            sil_score_max = sil_score
            ClusterNum = n_clusters
            Cluster_text = "Silhouette Score calculated %i clusters" %(ClusterNum)
        showinfo(title="Results", message=Cluster_text)
        
def RunKM_clustering():

    global df, df_nan, df_r, df_output, ClusterNum

    if df is None:
        showwarning("Warning", "Read file first")
    elif df_nan is None:
        showwarning("Warning", "Replace blank values first")
    elif df_r is None:
        showwarning("WARNING", "Clean file first")
    elif df_output is not None:
        showwarning("WARNING", "Already computed KMeans clusters")
    else:       
        df_output = df
        kmeans = cluster.KMeans(n_jobs = 8, n_clusters = ClusterNum, init = 'k-means++', random_state = RANDOM_STATE).fit(df_r)

        centroids = kmeans.cluster_centers_
        dists = pd.DataFrame(
            sdist.cdist(df_r, centroids), 
            columns=['dist_{}'.format(i) for i in range(len(centroids))],
            index=df_output.index)
        df_output = pd.concat([df_r, dists], axis=1)
        
        df_output['cluster_number'] = kmeans.labels_
        
        df_output_columns = df_output.filter(regex=('dist_')).columns

        df_output['dists'] = df_output[df_output_columns].min(axis=1)

        df_output = df_output[df_output.columns.drop(list(df_output.filter(regex='dist_')))]

        showinfo("INFO", "KMeans clusters calculated")

def RunKM_outlier():

    global df, df_nan, df_r, df_output
    
    if df is None:
        showwarning("Warning", "Read file first")
    elif df_nan is None:
        showwarning("Warning", "Replace blank values first")
    elif df_r is None:
        showwarning("WARNING", "Clean file first")
    elif df_output is not None:
        kmeans_one = cluster.KMeans(n_jobs = 8, n_clusters = 1, init = 'k-means++', random_state = RANDOM_STATE).fit(df_r)

        centroids = kmeans_one.cluster_centers_
        dist = pd.DataFrame(
            sdist.cdist(df_r, centroids), 
            columns=['dist_clust_outlier'.format(i) for i in range(len(centroids))],
            index=df_output.index)
        df_output = pd.concat([df_output, dist], axis=1)
        showinfo("INFO", "KMeans Outlier-cluster calculated and added to output \n Values represent distance from centroid")
    else:    
        df_output = df
        kmeans_one = cluster.KMeans(n_jobs = 8, n_clusters = 1, init = 'k-means++', random_state = RANDOM_STATE).fit(df_r)

        centroids = kmeans_one.cluster_centers_
        dist = pd.DataFrame(
            sdist.cdist(df_r, centroids), 
            columns=['dist_clust_outlier'.format(i) for i in range(len(centroids))],
            index=df_output.index)
        df_output = pd.concat([df_output, dist], axis=1)
        showinfo("INFO", "KMeans Outlier-cluster calculated \nValues represent distance from centroid")

def RunIsoForest():
    
    global df, df_nan, df_r, df_output, ContNum
    
    if df is None:
        showwarning("Warning", "Read file first")
    elif df_nan is None:
        showwarning("Warning", "Replace blank values first")
    elif df_r is None:
        showwarning("WARNING", "Clean file first")
    elif ContNum is None:
        showwarning('Warning', "Define Contamination Value")
    elif df_output is not None:
        iso_columns = df_r.columns
        clf = IsolationForest(max_samples="auto", n_jobs=-1, random_state=RANDOM_STATE, behaviour="new", contamination=ContNum)

        clf.fit(df_r[iso_columns])

        iso_pred = clf.predict(df_r[iso_columns])

        df_output['isolation_class'] = iso_pred
        showinfo("INFO", "IsolationForest Calculated")
    else:  
        df_output = df
        
        iso_columns = df_r.columns
        clf = IsolationForest(max_samples="auto", n_jobs=-1, random_state=RANDOM_STATE, behaviour="new", contamination=ContNum)

        clf.fit(df_r[iso_columns])

        iso_pred = clf.predict(df_r[iso_columns])

        df_output['isolation_class'] = iso_pred
        showinfo("INFO", "IsolationForest calculated")

def RunGB_Classifier():
    
    global df, df_r, df_nan, GB_Classifier_score, df_output
    
    if df is None:
        showwarning("Warning", "Read file first")
    elif df_nan is None:
        showwarning("Warning", "Replace blank values first")
    elif df_r is None:
        showwarning("WARNING", "Clean file first")
    else:
        df_output = df

        X = df_r.drop(df_r.columns[-1],axis=1)
        y = df_r.iloc[:,-1]

        cols = [c for c in df_r.columns]
        y_name = cols[-1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

        clf = GradientBoostingClassifier()

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        GB_Classifier_score = (accuracy_score(y_test, y_pred))

        full_pred = clf.predict(X)
        df_output['gb_predictions'] = full_pred

        showinfo(title="Accuracy score", message=GB_Classifier_score)
    
        
def save_in_new_file():

    global df, df_nan, df_r, df_output

    if df is None:
        showwarning("Warning", "Read file first")
    elif df_nan is None:
        showwarning("Warning", "Replace blank values first")
    elif df_r is None:
        showwarning("WARNING", "Clean file first")
    elif df_output is None:
        showwarning("WARNING", "Compute file first")
    else:
        df_output.to_csv("output.csv", sep=';', index=0, mode='w')
        showinfo("INFO", "DataFrame saved")
        
def ClusterDefinition():
    
    global ClusterNum
    
    ClusterNum = Clus_Number.get()
    ClusterNum = int(ClusterNum)
    print(ClusterNum)
    Cluster_text = "KMeans clustering set to %i clusters" %(ClusterNum)
    showinfo(title="Results", message=Cluster_text)

def ContaminationDefinition():
    
    global ContNum
    
    ContNum = Cont_Number.get()
    ContNum = int(ContNum)
    ContNum = ContNum/100 # to return integer to 0.0X state
    print(ContNum)
    Contamination_text = "Contamination value set to %0.2f" %(ContNum)
    showinfo(title="Results", message=Contamination_text)

def DisplayDF():
    
    global df
    
    if df is None:
        showwarning("Warning", "Read file first")
    else:
        win = Toplevel()
        message = "Current input dataframe (Max 100 rows)"
        tk.Label(win, text=message).pack()
        text = tk.Text(win)
        text.insert(tk.END, str(df.head(100)))
        text.pack()
        
def DisplayDF_R():
    
    global df
    
    if df is None:
        showwarning("Warning", "Read file first")
    else:
        win = Toplevel()
        message = "Current transformed dataframe (Max 100 rows)"
        tk.Label(win, text=message).pack()
        text = tk.Text(win)
        text.insert(tk.END, str(df_r.head(100)))
        text.pack()

def DisplayOutput():
    
    global df_output
    
    if df_output is None:
        showwarning("Warning", "Create output first")
    else:
        win = Toplevel()
        message = "Current output dataframe (Max 100 rows)"
        tk.Label(win, text=message).pack()
        text = tk.Text(win)
        text.insert(tk.END, str(df_output.head(100)))
        text.pack()


        
'''

┏━━━━•❍•°•❍•°•❍•━━━━┓
❍       TKINTER PAGES       ❍
┗━━━━•❍•°•❍•°•❍•━━━━┛


'''

class Analyticsapp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        #tk.Tk.iconbitmap(self, default="iconimage_kmeans.ico") #Icon for program
        tk.Tk.wm_title(self, "Advanched analytics")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        menubar = tk.Menu(container)
        filemenu_file = tk.Menu(menubar, tearoff=0)
        filemenu_file.add_command(label="Open", command=open_file)
        filemenu_file.add_command(label="Save", command=save_in_new_file)
        filemenu_file.add_separator()
        filemenu_file.add_command(label="Exit", command=Credits)
        menubar.add_cascade(label="File", menu=filemenu_file)
        
        filemenu_df = tk.Menu(menubar, tearoff=0)
        filemenu_df.add_command(label="Display input dataframe", command=DisplayDF)
        filemenu_df.add_command(label="Display transformed dataframe", command=DisplayDF_R)
        filemenu_df.add_command(label="Display output dataframe", command=DisplayOutput)
        menubar.add_cascade(label="Dataframes", menu=filemenu_df)
        
        filemenu_edit = tk.Menu(menubar, tearoff=0)
        submenu_t = tk.Menu(container, tearoff=0)
        submenu_s = tk.Menu(container, tearoff=0)
        submenu_t.add_command(label="Blank values", command=NaNifier)
        submenu_t.add_command(label="Non-numeric values", command=data_remover)
        submenu_s.add_command(label="MinMax scaler", command=MM_data_transformer)
        submenu_s.add_command(label="Standard scaler", command=S_data_transformer)
        filemenu_edit.add_cascade(label='Data transformating', menu=submenu_t, underline=0)
        filemenu_edit.add_cascade(label='Data scaling', menu=submenu_s, underline=0)
        filemenu_edit.add_separator()
        filemenu_edit.add_command(label="MinMax automatic data cleaning", command=AutoCleaner_MM)
        filemenu_edit.add_command(label="StandardScalar automatic data cleaning", command=AutoCleaner_S)
        filemenu_edit.add_separator()
        filemenu_edit.add_command(label="Clear input dataframe", command=ResetterDF)
        menubar.add_cascade(label="Edit", menu=filemenu_edit)    
        
        filemenu_help = tk.Menu(menubar, tearoff=0)
        filemenu_help.add_command(label="Function descriptions", command=RunUC) # Make help txt. file
        filemenu_help.add_command(label="Data cleaning steps", command=RunUC)
        filemenu_help.add_separator()
        filemenu_help.add_command(label="Credits", command=Credits)
        menubar.add_cascade(label="Help", menu=filemenu_help)
        
        tk.Tk.config(self, menu=menubar)
        
        self.frames = {}

        for F in (StartPage, ClusterPage, OutlierPage, ClassifierPage, PlottingDFPage, SystemPage, ElbowPage):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.graph_cpu = GraphPage_cpu(self, nb_points=360)
        self.graph_cpu.withdraw()  # hide window
        
        self.graph_mem = GraphPage_mem(self, nb_points=360)
        self.graph_mem.withdraw()  # hide window

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

    def show_graph_cpu(self):
        self.graph_cpu.deiconify()
        
    def show_graph_mem(self):
        self.graph_mem.deiconify()

class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Advanched analytics", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Clustering", 
                            command=lambda: controller.show_frame(ClusterPage))
        button1.pack(fill='x')
        
        button2 = ttk.Button(self, text="Outlier Detection", 
                            command=lambda: controller.show_frame(OutlierPage))
        button2.pack(fill='x')
        
        button3 = ttk.Button(self, text="Classification", 
                            command=lambda: controller.show_frame(ClassifierPage))
        button3.pack(fill='x')
        
        button4 = ttk.Button(self, text="Plot Example", 
                            command=lambda: controller.show_frame(PlottingDFPage))
        button4.pack(fill='x')
        
        button5 = ttk.Button(self, text="System Monitor", 
                            command=lambda: controller.show_frame(SystemPage))
        button5.pack(fill='x')

class ClusterPage(tk.Frame):    

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Clustering", font=LARGE_FONT)
        label.pack(pady=10, padx=10) 
        
        button1 = ttk.Button(self, text='Run Silhouette Score Calculation', command=RunSilHou)
        button1.pack(fill='x')
        
        button2 = ttk.Button(self, text="Elbow Method Calculation",
                             command=lambda: controller.show_frame(ElbowPage))
        button2.pack(fill='x')
        
        button3 = ttk.Button(self, text='Run KMeans', command=RunKM_clustering)
        button3.pack(fill='x')
        
        button4 = ttk.Button(self, text="Back",
                             command=lambda: controller.show_frame(StartPage))
        button4.pack(fill='x') 

class ElbowPage(tk.Frame):    

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Elbow Method", font=LARGE_FONT)
        label.pack(pady=10, padx=10) 
        
        global ClusterNum, C_Number

        def RunElbowMethod():
            
            global df, df_nan, df_r, df_output

            if df is None:
                showwarning("Warning", "Read file first")
            elif df_nan is None:
                showwarning("Warning", "Replace blank values first")
            elif df_r is None:
                showwarning("WARNING", "Clean file first")
            else:
                f = None
                f = plt.figure(figsize=(10, 8))
                wcss = []
                for i in range(1, 11):
                    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = RANDOM_STATE)
                    kmeans.fit(df)
                    wcss.append(kmeans.inertia_)
                plt.plot(range(1, 11), wcss)
                plt.title('The Elbow Method')
                plt.xlabel('Number of clusters')
                plt.ylabel('WCSS')
                plt.draw()

                canvas = FigureCanvasTkAgg(f, self)
                canvas.draw()
                canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

                toolbar = NavigationToolbar2Tk(canvas, self)
                toolbar.update()
                canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        ClusterOptions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        Clus_Number = StringVar(self)
        Clus_Number.set(ClusterOptions[0]) # default value

        button1 = ttk.Button(self, text="Run Elbow Method", command=RunElbowMethod)
        button1.pack(fill='x')
        
        dropdown1 = tk.OptionMenu(self, Clus_Number, *ClusterOptions)
        dropdown1.pack()
            
        button2 = ttk.Button(self, text="Define Number of Clusters", command=ClusterDefinition)
        button2.pack(fill='x')
        
        button3 = ttk.Button(self, text="Back",
                           command=lambda: controller.show_frame(ClusterPage))
        button3.pack(fill='x') 
        
class OutlierPage(tk.Frame):    

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Outlier Detection", font=LARGE_FONT)
        label.pack(pady=10, padx=10)            
        
        global ContNum, Cont_Number
        
        ContOptions = np.arange(1, 100, 1)
        Cont_Number = StringVar(self)
        Cont_Number.set(ContOptions[0]) # default value
        
        dropdown1 = tk.OptionMenu(self, Cont_Number, *ContOptions)
        dropdown1.pack()
            
        button2 = ttk.Button(self, text="Define Contamination Value (0.X Decimals)", command=ContaminationDefinition)
        button2.pack(fill='x')
        
        button3 = ttk.Button(self, text='Run IsolationForest Outlier Detection', command=RunIsoForest)
        button3.pack(fill='x')
        
        button4 = ttk.Button(self, text='Run KMeans Outlier Detection', command=RunKM_outlier)
        button4.pack(fill='x')
        
        button5 = ttk.Button(self, text="Back",
                           command=lambda: controller.show_frame(StartPage))
        button5.pack(fill='x')

class ClassifierPage(tk.Frame):    

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Outlier Detection", font=LARGE_FONT)
        label.pack(pady=10, padx=10) 
        
        button1 = ttk.Button(self, text='Run GB Classification', command=RunGB_Classifier)
        button1.pack(fill='x')
        
        button2 = ttk.Button(self, text="Back",
                           command=lambda: controller.show_frame(StartPage))
        button2.pack(fill='x')
        
class SystemPage(tk.Frame):    

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Outlier Detection", font=LARGE_FONT)
        label.pack(pady=10, padx=10) 

        button1 = ttk.Button(self, text="CPU Usage",
                             command=controller.show_graph_cpu)
        button1.pack(fill='x')
        
        button2 = ttk.Button(self, text="Memory Usage",
                             command=controller.show_graph_mem)
        button2.pack(fill='x')
        
        button3 = ttk.Button(self, text="Back",
                           command=lambda: controller.show_frame(StartPage))
        button3.pack(fill='x')
        
class GraphPage_cpu(tk.Toplevel):

    def __init__(self, parent, nb_points=360):
        tk.Toplevel.__init__(self, parent)
        self.protocol('WM_DELETE_WINDOW', self.withdraw)  # make the close button in the titlebar withdraw the toplevel instead of destroying it
        label = tk.Label(self, text="CPU Usage", font=LARGE_FONT)
        label.pack(pady=10, padx=10, side='top')

        # matplotlib figure
        self.figure = Figure(figsize=(5, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        # format the x-axis to show the time
        myFmt = mdates.DateFormatter("%H:%M:%S")
        self.ax.xaxis.set_major_formatter(myFmt)
        # initial x and y data
        dateTimeObj = datetime.now() + timedelta(seconds=-nb_points)
        self.x_data = [dateTimeObj + timedelta(seconds=i) for i in range(nb_points)]
        self.y_data = [0 for i in range(nb_points)]
        # create the plot
        self.plot = self.ax.plot(self.x_data, self.y_data, label='CPU')[0]
        self.ax.set_ylim(0, 100)
        self.ax.set_xlim(self.x_data[0], self.x_data[-1])

        self.canvas = FigureCanvasTkAgg(self.figure, self)

        toolbar = NavigationToolbar2Tk(self.canvas, self)
        toolbar.update()

        button1 = ttk.Button(self, text="Close", command=self.withdraw)
        button1.pack(side='bottom')

        self.canvas.get_tk_widget().pack(side='top', fill=tk.BOTH, expand=True)
        self.animate_cpu()

    def animate_cpu(self):
        # append new data point to the x and y data
        self.x_data.append(datetime.now())
        self.y_data.append(cpu_percent())
        # remove oldest data point
        self.x_data = self.x_data[1:]
        self.y_data = self.y_data[1:]
        #  update plot data
        self.plot.set_xdata(self.x_data)
        self.plot.set_ydata(self.y_data)
        self.ax.set_xlim(self.x_data[0], self.x_data[-1])
        self.canvas.draw_idle()  # redraw plot
        self.after(1000, self.animate_cpu)  # repeat after 1s

class GraphPage_mem(tk.Toplevel):

    def __init__(self, parent, nb_points=360):
        tk.Toplevel.__init__(self, parent)
        self.protocol('WM_DELETE_WINDOW', self.withdraw)  # make the close button in the titlebar withdraw the toplevel instead of destroying it
        label = tk.Label(self, text="Memory Usage", font=LARGE_FONT)
        label.pack(pady=10, padx=10, side='top')

        # matplotlib figure
        self.figure = Figure(figsize=(5, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        # format the x-axis to show the time
        myFmt = mdates.DateFormatter("%H:%M:%S")
        self.ax.xaxis.set_major_formatter(myFmt)
        # initial x and y data
        dateTimeObj = datetime.now() + timedelta(seconds=-nb_points)
        self.x_data = [dateTimeObj + timedelta(seconds=i) for i in range(nb_points)]
        self.y_data = [0 for i in range(nb_points)]
        # create the plot
        self.plot = self.ax.plot(self.x_data, self.y_data, label='Memory')[0]
        self.ax.set_ylim(0, 100)
        self.ax.set_xlim(self.x_data[0], self.x_data[-1])

        self.canvas = FigureCanvasTkAgg(self.figure, self)

        toolbar = NavigationToolbar2Tk(self.canvas, self)
        toolbar.update()

        button1 = ttk.Button(self, text="Close", command=self.withdraw)
        button1.pack(side='bottom')

        self.canvas.get_tk_widget().pack(side='top', fill=tk.BOTH, expand=True)
        self.animate_mem()
        
    def animate_mem(self):
        # append new data point to the x and y data
        self.x_data.append(datetime.now())
        self.y_data.append(virtual_memory().percent)
        # remove oldest data point
        self.x_data = self.x_data[1:]
        self.y_data = self.y_data[1:]
        #  update plot data
        self.plot.set_xdata(self.x_data)
        self.plot.set_ydata(self.y_data)
        self.ax.set_xlim(self.x_data[0], self.x_data[-1])
        self.canvas.draw_idle()  # redraw plot
        self.after(1000, self.animate_mem)  # repeat after 1s   
        
class PlottingDFPage(tk.Frame):    

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Plotting Dataframe", font=LARGE_FONT)
        label.pack(pady=10, padx=10)         

        global df, Plot_type, Plot_define
        
        def PlotMaster():

            Plot_define = Plot_type.get()

            if df is None:
                showwarning("Warning", "Read file first")
            elif Plot_define == '-Choose an option-':
                showwarning("Warning", "Choose a plot type")
            elif Plot_define == 'Histogram':
                Plot_text = "Plot set to: "+Plot_define
                showinfo(title="Results", message=Plot_text)
                HistogramDF()
            elif Plot_define == 'Scatterplot':
                Plot_text = "Plot set to: "+Plot_define
                showinfo(title="Results", message=Plot_text)
                ScatterplotDF()
            elif Plot_define == 'Boxplot':
                Plot_text = "Plot set to: "+Plot_define
                showinfo(title="Results", message=Plot_text)
                BoxplotDF()
            else: None
        
        def HistogramDF():
            
            f = None
            f = plt.figure(figsize=(10, 8))
            
            # REMOVE AFTER DF X, y generation!!!
            ax = sns.distplot(df['age'])

            canvas = FigureCanvasTkAgg(f, self)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

            toolbar = NavigationToolbar2Tk(canvas, self)
            toolbar.update()
            canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        def ScatterplotDF():
            
            f = None
            f = plt.figure(figsize=(10, 8))
            
            x = df['age']
            y = df['collection_agency']
            
            plt.scatter(x, y)
            plt.draw()
            
            canvas = FigureCanvasTkAgg(f, self)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

            toolbar = NavigationToolbar2Tk(canvas, self)
            toolbar.update()
            canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        def BoxplotDF():
            
            f = None
            f = plt.figure(figsize=(10, 8))
            
            # REMOVE AFTER DF X, y generation!!!
            boxplot_df = df.filter(['age', 'collection_agency'])
            boxplot_columns = list(boxplot_df.columns)
        
            sns.set_style('whitegrid')
            ax = sns.boxplot(x='collection_agency',y='age',data=boxplot_df)
            ax = sns.stripplot(x='collection_agency', y='age',data=boxplot_df)
            
            canvas = FigureCanvasTkAgg(f, self)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

            toolbar = NavigationToolbar2Tk(canvas, self)
            toolbar.update()
            canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        Options_plot = ['-Choose an option-', 'Histogram', 'Scatterplot', 'Boxplot']
                
        Plot_type = StringVar(self)
        Plot_type.set(Options_plot[0]) # default value
        
        label1 = tk.Label(self, text='Plot type:')
        label1.pack()
        
        dropdown1 = tk.OptionMenu(self, Plot_type, *Options_plot)
        dropdown1.pack()
        
        button1 = ttk.Button(self, text="Run plot", command=PlotMaster)
        button1.pack(fill='x')
        
        button2 = ttk.Button(self, text="Back",
                           command=lambda: controller.show_frame(StartPage))
        button2.pack(fill='x') 
        
app = Analyticsapp()
app.geometry('500x400')
app.mainloop()
