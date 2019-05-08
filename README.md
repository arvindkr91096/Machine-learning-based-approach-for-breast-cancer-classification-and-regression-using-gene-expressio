ARVIND KUMAR M.SC BIOINFORMATICS

what is the machine learning
A machine learning is a type of Artificial Intelligency that allows software applications to learn from the data and become more accurate in predicting outcomes without human intervention.

Training Data => Learning Algorithm => Build Model => Perform => Feedback

Type of machine learning
1.Supervised Learning 2.unsupervised

1.Supervised Learning:- This is a process of an algorithm learning from the training dataset.

2.unsupervised:- This is a process where a model is trained using an information which is not labelled.

Type of supervised Learning
There are 2 Type of supervised Learning 1.Regression 2. Classification.

1.Regression:- Regression is the prediction of a numeric value and often taken input as a continuous value.

2.Classification:-Classification is the problem identifying to which set of categories a new observation belong

Type of Regression Algorithm
1.Linear Regression
image.png

2.Logistic Regression
image.png

Type of Classification Algorithm
1.k-Nearest neighbor(KNN) 2.Support Vector Machine(svm) 3.Naive Bayes 4.Decision Tree 5.Random Forest6.Neural Network

1.KNN:-
The K-nearest neighbors (KNN) algorithm is a type of supervised machine learning algorithms. KNN is extremely easy to implement in its most basic form, and yet performs quite complex classification tasks. It is a lazy learning algorithm since it doesn't have a specialized training phase. image.png

2.SVM:-
A Support Vector Machine (SVM) is a discriminative classifier formally defined by a separating hyperplane. In other words, given labeled training data (supervised learning), the algorithm outputs an optimal hyperplane which categorizes new examples. image.png

3.Naive Bayes:-
The Naive Bayesian classifier is based on Bayes’ theorem with the independence assumptions between predictors. A Naive Bayesian model is easy to build, with no complicated iterative parameter estimation which makes it particularly useful for very large datasets. Despite its simplicity, the Naive Bayesian classifier often does surprisingly well and is widely used because it often outperforms more sophisticated classification methods. image.png

4.Decision Tree:-
A decision tree is a decision support tool that uses a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility. It is one way to display an algorithm that only contains conditional control statements. image.png

5.Random Forest:-
Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.Random decision forests correct for decision trees' habit of overfitting to their training set image.png

6.Neural Network:-
a computer system modelled on the human brain and nervous system. image.png

Installing Libraries
1.python3.6:-
Python is an interpreted, high-level, general-purpose programming language.it is easy to learn,write and execute.installation in ubuntus 18.04 (sudo apt-get install python3.6)

2.jupyter notebook :-
it is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text. Uses include: data cleaning and transformation, numerical simulation, statistical modeling, data visualization, machine learning, and much more.installation in ubuntus18.04(pip install jupyter notebook).

2.scikitlearn(sklearn):-
it is a free software machine learning library for the Python programming language.It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.installation in python3.6(pip install sklearn).

3.matplotlib:-
Matplotlib is a python library used to create 2D graphs and plots by using python scripts. It has a module named pyplot which makes things easy for plotting by providing feature to control line styles, font properties, formatting axes etc. It supports a very wide variety of graphs and plots namely - histogram, bar charts, power spectra, error charts etc. It is used along with NumPy to provide an environment that is an effective open source alternative for MatLab.installation in python3.6(pip install matplotlib)

pandas:-
it is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.installation in python3.6(pip install pandas)

sciborn:-
it is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.

DATASET
This data are provied in GEO datasets.accession no is GSM50948

feature:-
There are 2 Type of Feature (1)categorical Feature:-all alphabate ex-gene_name,geneid(2)continuous Feature:-all numerical value ex-gene expression. in this dataset univarince categorical data and multivariance continuous data available here.There are 2 Type of variable(1)dependent variable (2)independent variable.Feature is a independent variable but label is a dependent variable.

Label:-
it is the output given input.it is the dependent variable.

Summary:-
These data can be used for evaluation of the clinical utility of the research-based PAM50 subtype predictor in predicting pathological complete response (pCR) and event-free survival (EFS) in women enrolled in the NeOAdjuvant Herceptin (NOAH) trial. The NeOAdjuvant Herceptin [NOAH] trial demonstrated that trastuzumab significantly improves pCR rates and 3-year event-free survival (EFS) in combination with neoadjuvant chemotherapy compared with neoadjuvant chemotherapy alone in patients with HER2+ breast cancer.

Overall :-
Gene expression profiling was performed using RNA from formalin-fixed paraffin-embedded core biopsies from 114 pre-treated patients with HER2-positive (HER2+) tumors randomized to receive neoadjuvant doxorubicin/paclitaxel (AT) followed by cyclophosphamide/methotrexate/fluorouracil (CMF), or the same regimen in combination with trastuzumab for 1 year. A control cohort of 42 patients with HER2-negative tumors treated with AT-CMF was also included.This dataset represents 156 patients who had provided the samples at baseline (pre-treatment).

 
# here we will import the libraries used for machine learning
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph. I like it most for plot
%matplotlib inline
from sklearn.linear_model import LogisticRegression # to apply the Logistic regression
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn import datasets
from sklearn.model_selection import GroupKFold
# use for cross validation
from sklearn.model_selection import GridSearchCV# for tuning parameter
from sklearn.ensemble import RandomForestClassifier # for random forest classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import linear_model
#from sklearn.cross_validation import KFold
from sklearn import svm
from sklearn.svm import SVC# for Support Vector Machine
from sklearn import metrics # for the check the error and accuracy of the model
# Any results you write to the current directory are saved as output.
# dont worry about the error if its not working then insteda of model_selection we can use cross_validation
data = pd.read_csv("GSE50948_series_matrix1.csv")
print(data.head(2))
  Type  GSM1232992  GSM1232993  GSM1232994  GSM1232995  GSM1232996  \
0  ER-    6.777140    6.781371    6.387051    6.870976    6.614293   
1  ER-    1.680005    1.795889    1.688350    1.673858    1.588154   

   GSM1232997  GSM1232998  GSM1232999  GSM1233000  ...  GSM1233138  \
0    6.961617    6.211377    6.470648    7.770163  ...    5.338288   
1    1.554756    1.611814    1.329927    1.644885  ...    1.726601   

   GSM1233139  GSM1233140  GSM1233141  GSM1233142  GSM1233143  GSM1233144  \
0    7.368802    6.649013    5.782790    6.546996    6.806370    7.461399   
1    1.376361    2.074658    1.575916    1.709130    1.754602    1.816113   

   GSM1233145  GSM1233146  GSM1233147  
0    5.840764    7.342973    6.781235  
1    1.910038    1.384827    1.705504  

[2 rows x 157 columns]
data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 54675 entries, 0 to 54674
Columns: 157 entries, Type to GSM1233147
dtypes: float64(156), object(1)
memory usage: 65.5+ MB
data.columns
Index(['Type', 'GSM1232992', 'GSM1232993', 'GSM1232994', 'GSM1232995',
       'GSM1232996', 'GSM1232997', 'GSM1232998', 'GSM1232999', 'GSM1233000',
       ...
       'GSM1233138', 'GSM1233139', 'GSM1233140', 'GSM1233141', 'GSM1233142',
       'GSM1233143', 'GSM1233144', 'GSM1233145', 'GSM1233146', 'GSM1233147'],
      dtype='object', length=157)
#data.drop("type",axis=1,inplace=True)
data=data.transpose()
features_mean= list(data.columns[1:30])
features_se= list(data.columns[31:60])
features_worst=list(data.columns[61:90])
features_mean1=list(data.columns[91:120])
features_se1=list(data.columns[121:157])
print(features_mean)
print("-----------------------------------")
print(features_se)
print("------------------------------------")
print(features_worst)
print("------------------------------------")
print(features_mean1)
print("------------------------------------")
print(features_se1)
print("------------------------------------")
['GSM1232992', 'GSM1232993', 'GSM1232994', 'GSM1232995', 'GSM1232996', 'GSM1232997', 'GSM1232998', 'GSM1232999', 'GSM1233000', 'GSM1233001', 'GSM1233002', 'GSM1233003', 'GSM1233004', 'GSM1233005', 'GSM1233006', 'GSM1233007', 'GSM1233008', 'GSM1233009', 'GSM1233010', 'GSM1233011', 'GSM1233012', 'GSM1233013', 'GSM1233014', 'GSM1233015', 'GSM1233016', 'GSM1233017', 'GSM1233018', 'GSM1233019', 'GSM1233020']
-----------------------------------
['GSM1233022', 'GSM1233023', 'GSM1233024', 'GSM1233025', 'GSM1233026', 'GSM1233027', 'GSM1233028', 'GSM1233029', 'GSM1233030', 'GSM1233031', 'GSM1233032', 'GSM1233033', 'GSM1233034', 'GSM1233035', 'GSM1233036', 'GSM1233037', 'GSM1233038', 'GSM1233039', 'GSM1233040', 'GSM1233041', 'GSM1233042', 'GSM1233043', 'GSM1233044', 'GSM1233045', 'GSM1233046', 'GSM1233047', 'GSM1233048', 'GSM1233049', 'GSM1233050']
------------------------------------
['GSM1233052', 'GSM1233053', 'GSM1233054', 'GSM1233055', 'GSM1233056', 'GSM1233057', 'GSM1233058', 'GSM1233059', 'GSM1233060', 'GSM1233061', 'GSM1233062', 'GSM1233063', 'GSM1233064', 'GSM1233065', 'GSM1233066', 'GSM1233067', 'GSM1233068', 'GSM1233069', 'GSM1233070', 'GSM1233071', 'GSM1233072', 'GSM1233073', 'GSM1233074', 'GSM1233075', 'GSM1233076', 'GSM1233077', 'GSM1233078', 'GSM1233079', 'GSM1233080']
------------------------------------
['GSM1233082', 'GSM1233083', 'GSM1233084', 'GSM1233085', 'GSM1233086', 'GSM1233087', 'GSM1233088', 'GSM1233089', 'GSM1233090', 'GSM1233091', 'GSM1233092', 'GSM1233093', 'GSM1233094', 'GSM1233095', 'GSM1233096', 'GSM1233097', 'GSM1233098', 'GSM1233099', 'GSM1233100', 'GSM1233101', 'GSM1233102', 'GSM1233103', 'GSM1233104', 'GSM1233105', 'GSM1233106', 'GSM1233107', 'GSM1233108', 'GSM1233109', 'GSM1233110']
------------------------------------
['GSM1233112', 'GSM1233113', 'GSM1233114', 'GSM1233115', 'GSM1233116', 'GSM1233117', 'GSM1233118', 'GSM1233119', 'GSM1233120', 'GSM1233121', 'GSM1233122', 'GSM1233123', 'GSM1233124', 'GSM1233125', 'GSM1233126', 'GSM1233127', 'GSM1233128', 'GSM1233129', 'GSM1233130', 'GSM1233131', 'GSM1233132', 'GSM1233133', 'GSM1233134', 'GSM1233135', 'GSM1233136', 'GSM1233137', 'GSM1233138', 'GSM1233139', 'GSM1233140', 'GSM1233141', 'GSM1233142', 'GSM1233143', 'GSM1233144', 'GSM1233145', 'GSM1233146', 'GSM1233147']
------------------------------------
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in data.columns:
    if data[col].dtypes == 'object':
        data[col] = le.fit_transform(data[col])
#import pandas as pd
#data['type']=data['type'].map({'M':1,'B':0})
data.describe()
Type	GSM1232992	GSM1232993	GSM1232994	GSM1232995	GSM1232996	GSM1232997	GSM1232998	GSM1232999	GSM1233000	...	GSM1233138	GSM1233139	GSM1233140	GSM1233141	GSM1233142	GSM1233143	GSM1233144	GSM1233145	GSM1233146	GSM1233147
count	54675.000000	54675.000000	54675.000000	54675.000000	54675.000000	54675.000000	54675.000000	54675.000000	54675.000000	54675.000000	...	54675.000000	54675.000000	54675.000000	54675.000000	54675.000000	54675.000000	54675.000000	54675.000000	54675.000000	54675.000000
mean	0.686036	3.738098	3.742048	3.738089	3.732380	3.733062	3.738695	3.734469	3.735414	3.735784	...	3.727533	3.736979	3.755305	3.739884	3.729446	3.735635	3.736039	3.770564	3.741893	3.741309
std	0.490993	1.817426	1.759053	1.790047	1.855668	1.824573	1.754655	1.798407	1.766887	1.803805	...	1.723070	1.803777	1.634157	1.808187	1.828128	1.755436	1.677219	1.612947	1.698765	1.731300
min	0.000000	0.675627	0.661950	0.726501	0.657045	0.707924	0.740344	0.647055	0.677520	0.633358	...	0.700096	0.654189	0.683605	0.616789	0.652717	0.637530	0.704062	0.631458	0.650578	0.649511
25%	0.000000	2.287240	2.325594	2.311779	2.226626	2.269879	2.323323	2.292703	2.308267	2.280783	...	2.358481	2.279692	2.450874	2.298311	2.258591	2.331571	2.391223	2.487880	2.382225	2.357980
50%	1.000000	3.543338	3.550937	3.524754	3.501463	3.522276	3.544673	3.536348	3.560477	3.550615	...	3.564324	3.559508	3.633955	3.537351	3.513523	3.571007	3.585198	3.649264	3.579700	3.566345
75%	1.000000	4.878133	4.819917	4.835364	4.884320	4.865119	4.825174	4.848346	4.846280	4.878982	...	4.822622	4.883532	4.820308	4.860755	4.858400	4.850093	4.813641	4.814813	4.826963	4.838565
max	2.000000	13.661864	13.692730	13.642381	13.635918	13.646637	13.689604	13.673741	13.651787	13.668725	...	13.359328	13.602657	13.425517	13.601234	13.516233	13.490688	13.776826	13.688906	13.799087	13.733208
8 rows × 157 columns

sns.countplot(data['Type'],label="Count")
<matplotlib.axes._subplots.AxesSubplot at 0x7ffa954f36a0>

corr = data[features_mean].corr() # .corr is used for find corelation
plt.figure(figsize=(20,20))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
           xticklabels= features_mean, yticklabels= features_mean,
           cmap= 'coolwarm')
<matplotlib.axes._subplots.AxesSubplot at 0x7ffa954de470>

In this fig:-the GSM1232992,GSM1232994,GSM1232996,GSM1232999 and GSM123233000 are highly correlated as expected from their relation so from these we will use anyone of them

prediction_var = ['GSM1232992','GSM1232993','GSM1232994','GSM1232995','GSM1232996','GSM1232997','GSM1232998','GSM1232999','GSM1233000','GSM1233001','GSM1233002','GSM1233003','GSM1233004','GSM1233005','GSM1233006','GSM1233007','GSM1233008','GSM1233009','GSM1233010','GSM1233011','GSM1233012','GSM1233013','GSM1233014','GSM1233015','GSM1233016','GSM1233017','GSM1233018','GSM1233019','GSM1233020','GSM1233021','GSM1233022','GSM1233023','GSM1233024','GSM1233025','GSM1233026','GSM1233027','GSM1233028','GSM1233029','GSM1233030','GSM1233031','GSM1233032','GSM1233033','GSM1233034','GSM1233035','GSM1233036','GSM1233037','GSM1233038','GSM1233039','GSM1233040','GSM1233041','GSM1233042','GSM1233043','GSM1233044','GSM1233012','GSM1233046','GSM1233047','GSM1233048','GSM1233049','GSM1233050','GSM1233051','GSM1233052','GSM1233053','GSM1233054','GSM1233055','GSM1233056','GSM1233057','GSM1233058','GSM1233059','GSM1233060','GSM1233061','GSM1233062','GSM1233063','GSM1233064','GSM1233065','GSM1233066','GSM1233067','GSM1233068','GSM1233069','GSM1233070','GSM1233071','GSM1233072','GSM1233073','GSM1233074','GSM1233075','GSM1233076','GSM1233077','GSM1233078','GSM1233079','GSM1233080','GSM1233081','GSM1233082','GSM1233083','GSM1233084','GSM1233085','GSM1233086','GSM1233087','GSM1233088','GSM1233089','GSM1233090','GSM1233091','GSM1233092','GSM1233093','GSM1233094','GSM1233095','GSM1233096','GSM1233097','GSM1233098','GSM1233099','GSM1233100','GSM1233101','GSM1233102','GSM1233103','GSM1233104','GSM1233105','GSM1233106','GSM1233107','GSM1233108','GSM1233109','GSM1233110','GSM1233111','GSM1233112','GSM1233113','GSM1233114','GSM1233115','GSM1233116','GSM1233117','GSM1233118','GSM1233119','GSM1233120','GSM1233121','GSM1233122','GSM1233123','GSM1233124','GSM1233125','GSM1233126','GSM1233127','GSM1233128','GSM1233129','GSM1233130','GSM1233131','GSM1233132','GSM1233133','GSM1233134','GSM1233135','GSM1233136','GSM1233137','GSM1233138','GSM1233139','GSM1233140','GSM1233141','GSM1233142','GSM1233143','GSM1233144','GSM1233145','GSM1233146','GSM1233147']
train, test = train_test_split(data, test_size = 0.3)
print(train.shape)
print(test.shape)
(38272, 157)
(16403, 157)
train_X = train[prediction_var]
train_y=train.Type
test_X= test[prediction_var]
test_y =test.Type
model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_y)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
prediction=model.predict(test_X)
metrics.accuracy_score(prediction,test_y)
0.6570749253185393
model = svm.SVC()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
metrics.accuracy_score(prediction,test_y)
/home/student/.local/lib/python3.5/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
0.6607327927818082
prediction_var = features_mean
train_X= train[prediction_var]
train_y= train.Type
test_X = test[prediction_var]
test_y = test.Type
model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_y)
prediction = model.predict(test_X)
metrics.accuracy_score(prediction,test_y)
0.6576236054380297
featimp = pd.Series(model.feature_importances_, index=prediction_var).sort_values(ascending=False)
print(featimp)
GSM1233014    0.035320
GSM1233004    0.035251
GSM1233003    0.035108
GSM1233008    0.034935
GSM1233002    0.034855
GSM1232998    0.034781
GSM1233005    0.034717
GSM1233020    0.034707
GSM1232994    0.034698
GSM1232995    0.034680
GSM1232997    0.034679
GSM1232996    0.034660
GSM1232993    0.034634
GSM1233012    0.034608
GSM1233013    0.034607
GSM1233010    0.034394
GSM1233019    0.034350
GSM1233001    0.034271
GSM1233016    0.034226
GSM1233011    0.034214
GSM1232999    0.034212
GSM1233006    0.034200
GSM1233015    0.034173
GSM1233007    0.034126
GSM1233017    0.034125
GSM1233009    0.034079
GSM1233018    0.033929
GSM1233000    0.033854
GSM1232992    0.033607
dtype: float64
model = svm.SVC()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
metrics.accuracy_score(prediction,test_y)
/home/student/.local/lib/python3.5/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
0.6606718283240871
prediction_var = ['GSM1232992','GSM1232993','GSM1232994','GSM1232995','GSM1232996','GSM1232997','GSM1232998','GSM1232999','GSM1233000','GSM1233001','GSM1233002','GSM1233003','GSM1233004','GSM1233005','GSM1233006','GSM1233007','GSM1233008','GSM1233009','GSM1233010','GSM1233011','GSM1233012','GSM1233013','GSM1233014','GSM1233015','GSM1233016','GSM1233017','GSM1233018','GSM1233019','GSM1233020','GSM1233021','GSM1233022','GSM1233023','GSM1233024','GSM1233025','GSM1233026','GSM1233027','GSM1233028','GSM1233029','GSM1233030','GSM1233031','GSM1233032','GSM1233033','GSM1233034','GSM1233035','GSM1233036','GSM1233037','GSM1233038','GSM1233039','GSM1233040','GSM1233041','GSM1233042','GSM1233043','GSM1233044','GSM1233012','GSM1233046','GSM1233047','GSM1233048','GSM1233049','GSM1233050','GSM1233051','GSM1233052','GSM1233053','GSM1233054','GSM1233055','GSM1233056','GSM1233057','GSM1233058','GSM1233059','GSM1233060','GSM1233061','GSM1233062','GSM1233063','GSM1233064','GSM1233065','GSM1233066','GSM1233067','GSM1233068','GSM1233069','GSM1233070','GSM1233071','GSM1233072','GSM1233073','GSM1233074','GSM1233075','GSM1233076','GSM1233077','GSM1233078','GSM1233079','GSM1233080','GSM1233081','GSM1233082','GSM1233083','GSM1233084','GSM1233085','GSM1233086','GSM1233087','GSM1233088','GSM1233089','GSM1233090','GSM1233091','GSM1233092','GSM1233093','GSM1233094','GSM1233095','GSM1233096','GSM1233097','GSM1233098','GSM1233099','GSM1233100','GSM1233101','GSM1233102','GSM1233103','GSM1233104','GSM1233105','GSM1233106','GSM1233107','GSM1233108','GSM1233109','GSM1233110','GSM1233111','GSM1233112','GSM1233113','GSM1233114','GSM1233115','GSM1233116','GSM1233117','GSM1233118','GSM1233119','GSM1233120','GSM1233121','GSM1233122','GSM1233123','GSM1233124','GSM1233125','GSM1233126','GSM1233127','GSM1233128','GSM1233129','GSM1233130','GSM1233131','GSM1233132','GSM1233133','GSM1233134','GSM1233135','GSM1233136','GSM1233137','GSM1233138','GSM1233139','GSM1233140','GSM1233141','GSM1233142','GSM1233143','GSM1233144','GSM1233145','GSM1233146','GSM1233147']
train_X= train[prediction_var]
train_y= train.Type
test_X = test[prediction_var]
test_y = test.Type
model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_y)
prediction = model.predict(test_X)
metrics.accuracy_score(prediction,test_y)
0.6560994939950009
model = svm.SVC()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
metrics.accuracy_score(prediction,test_y)
/home/student/.local/lib/python3.5/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
0.624946656099494
prediction_var = features_se
train_X= train[prediction_var]
train_y= train.Type
test_X = test[prediction_var]
test_y = test.Type
model.fit(train_X,train_y)
prediction = model.predict(test_X)
metrics.accuracy_score(prediction,test_y)
0.6553679205023472
model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_y)
prediction = model.predict(test_X)
metrics.accuracy_score(prediction,test_y)
0.656526245199049
featimp = pd.Series(model.feature_importances_, index=prediction_var).sort_values(ascending=False)
print(featimp)
GSM1233019    0.101367
GSM1233017    0.101095
GSM1233018    0.100625
GSM1233015    0.100407
GSM1233012    0.100102
GSM1233014    0.099950
GSM1233016    0.099742
GSM1233013    0.099537
GSM1233021    0.098874
GSM1233020    0.098301
dtype: float64
model = svm.SVC()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
metrics.accuracy_score(prediction,test_y)
/home/student/.local/lib/python3.5/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
0.6607327927818082
prediction_var = ['GSM1232992','GSM1232993','GSM1232994','GSM1232995','GSM1232996','GSM1232997','GSM1232998','GSM1232999','GSM1233000','GSM1233001','GSM1233002','GSM1233003','GSM1233004','GSM1233005','GSM1233006','GSM1233007','GSM1233008','GSM1233009','GSM1233010','GSM1233011','GSM1233012','GSM1233013','GSM1233014','GSM1233015','GSM1233016','GSM1233017','GSM1233018','GSM1233019','GSM1233020','GSM1233021','GSM1233022','GSM1233023','GSM1233024','GSM1233025','GSM1233026','GSM1233027','GSM1233028','GSM1233029','GSM1233030','GSM1233031','GSM1233032','GSM1233033','GSM1233034','GSM1233035','GSM1233036','GSM1233037','GSM1233038','GSM1233039','GSM1233040','GSM1233041','GSM1233042','GSM1233043','GSM1233044','GSM1233012','GSM1233046','GSM1233047','GSM1233048','GSM1233049','GSM1233050','GSM1233051','GSM1233052','GSM1233053','GSM1233054','GSM1233055','GSM1233056','GSM1233057','GSM1233058','GSM1233059','GSM1233060','GSM1233061','GSM1233062','GSM1233063','GSM1233064','GSM1233065','GSM1233066','GSM1233067','GSM1233068','GSM1233069','GSM1233070','GSM1233071','GSM1233072','GSM1233073','GSM1233074','GSM1233075','GSM1233076','GSM1233077','GSM1233078','GSM1233079','GSM1233080','GSM1233081','GSM1233082','GSM1233083','GSM1233084','GSM1233085','GSM1233086','GSM1233087','GSM1233088','GSM1233089','GSM1233090','GSM1233091','GSM1233092','GSM1233093','GSM1233094','GSM1233095','GSM1233096','GSM1233097','GSM1233098','GSM1233099','GSM1233100','GSM1233101','GSM1233102','GSM1233103','GSM1233104','GSM1233105','GSM1233106','GSM1233107','GSM1233108','GSM1233109','GSM1233110','GSM1233111','GSM1233112','GSM1233113','GSM1233114','GSM1233115','GSM1233116','GSM1233117','GSM1233118','GSM1233119','GSM1233120','GSM1233121','GSM1233122','GSM1233123','GSM1233124','GSM1233125','GSM1233126','GSM1233127','GSM1233128','GSM1233129','GSM1233130','GSM1233131','GSM1233132','GSM1233133','GSM1233134','GSM1233135','GSM1233136','GSM1233137','GSM1233138','GSM1233139','GSM1233140','GSM1233141','GSM1233142','GSM1233143','GSM1233144','GSM1233145','GSM1233146']
train_X= train[prediction_var]
train_y= train.Type
test_X = test[prediction_var]
test_y = test.Type
model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_y)
prediction = model.predict(test_X)
metrics.accuracy_score(prediction,test_y)
0.655306956044626
model = svm.SVC()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
metrics.accuracy_score(prediction,test_y)
/home/student/.local/lib/python3.5/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
0.6607327927818082
prediction_var = features_worst
train_X= train[prediction_var]
train_y= train.Type
test_X = test[prediction_var]
test_y = test.Type
model = svm.SVC()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
metrics.accuracy_score(prediction,test_y)
/home/student/.local/lib/python3.5/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
0.6607327927818082
model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_y)
prediction = model.predict(test_X)
metrics.accuracy_score(prediction,test_y)
0.6552459915869049
featimp = pd.Series(model.feature_importances_, index=prediction_var).sort_values(ascending=False)
print(featimp)
GSM1233057    0.035179
GSM1233066    0.035123
GSM1233056    0.035095
GSM1233053    0.035056
GSM1233079    0.034955
GSM1233062    0.034891
GSM1233069    0.034859
GSM1233055    0.034784
GSM1233059    0.034671
GSM1233080    0.034623
GSM1233072    0.034615
GSM1233064    0.034586
GSM1233054    0.034545
GSM1233063    0.034525
GSM1233075    0.034468
GSM1233060    0.034367
GSM1233061    0.034304
GSM1233071    0.034274
GSM1233077    0.034264
GSM1233068    0.034222
GSM1233052    0.034206
GSM1233074    0.034198
GSM1233073    0.034196
GSM1233078    0.034171
GSM1233067    0.034145
GSM1233058    0.034142
GSM1233076    0.033983
GSM1233070    0.033817
GSM1233065    0.033737
dtype: float64
Observation

features_mean
['GSM1232992',
 'GSM1232993',
 'GSM1232994',
 'GSM1232995',
 'GSM1232996',
 'GSM1232997',
 'GSM1232998',
 'GSM1232999',
 'GSM1233000',
 'GSM1233001',
 'GSM1233002',
 'GSM1233003',
 'GSM1233004',
 'GSM1233005',
 'GSM1233006',
 'GSM1233007',
 'GSM1233008',
 'GSM1233009',
 'GSM1233010',
 'GSM1233011',
 'GSM1233012',
 'GSM1233013',
 'GSM1233014',
 'GSM1233015',
 'GSM1233016',
 'GSM1233017',
 'GSM1233018',
 'GSM1233019',
 'GSM1233020']
prediction_var = ['GSM1232992','GSM1232993','GSM1232994','GSM1232995','GSM1232996','GSM1232997','GSM1232998','GSM1232999','GSM1233000','GSM1233001','GSM1233002','GSM1233003','GSM1233004','GSM1233005','GSM1233006','GSM1233007','GSM1233008','GSM1233009','GSM1233010','GSM1233011','GSM1233012','GSM1233013','GSM1233014','GSM1233015','GSM1233016','GSM1233017','GSM1233018','GSM1233019','GSM1233020','GSM1233021','GSM1233022','GSM1233023','GSM1233024','GSM1233025','GSM1233026','GSM1233027','GSM1233028','GSM1233029','GSM1233030','GSM1233031','GSM1233032','GSM1233033','GSM1233034','GSM1233035','GSM1233036','GSM1233037','GSM1233038','GSM1233039','GSM1233040','GSM1233041','GSM1233042','GSM1233043','GSM1233044','GSM1233012','GSM1233046','GSM1233047','GSM1233048','GSM1233049','GSM1233050','GSM1233051','GSM1233052','GSM1233053','GSM1233054','GSM1233055','GSM1233056','GSM1233057','GSM1233058','GSM1233059','GSM1233060','GSM1233061','GSM1233062','GSM1233063','GSM1233064','GSM1233065','GSM1233066','GSM1233067','GSM1233068','GSM1233069','GSM1233070','GSM1233071','GSM1233072','GSM1233073','GSM1233074','GSM1233075','GSM1233076','GSM1233077','GSM1233078','GSM1233079','GSM1233080','GSM1233081','GSM1233082','GSM1233083','GSM1233084','GSM1233085','GSM1233086','GSM1233087','GSM1233088','GSM1233089','GSM1233090','GSM1233091','GSM1233092','GSM1233093','GSM1233094','GSM1233095','GSM1233096','GSM1233097','GSM1233098','GSM1233099','GSM1233100','GSM1233101','GSM1233102','GSM1233103','GSM1233104','GSM1233105','GSM1233106','GSM1233107','GSM1233108','GSM1233109','GSM1233110','GSM1233111','GSM1233112','GSM1233113','GSM1233114','GSM1233115','GSM1233116','GSM1233117','GSM1233118','GSM1233119','GSM1233120','GSM1233121','GSM1233122','GSM1233123','GSM1233124','GSM1233125','GSM1233126','GSM1233127','GSM1233128','GSM1233129','GSM1233130','GSM1233131','GSM1233132','GSM1233133','GSM1233134','GSM1233135','GSM1233136','GSM1233137','GSM1233138','GSM1233139','GSM1233140']
def model(model,data,prediction,outcome):
     kf = KFold(data.shape[0], n_folds=10)
prediction_var = ['GSM1232992','GSM1232993','GSM1232994','GSM1232995','GSM1232996','GSM1232997','GSM1232998','GSM1232999','GSM1233000','GSM1233001','GSM1233002','GSM1233003','GSM1233004','GSM1233005','GSM1233006','GSM1233007','GSM1233008','GSM1233009','GSM1233010','GSM1233011','GSM1233012','GSM1233013','GSM1233014','GSM1233015','GSM1233016','GSM1233017','GSM1233018','GSM1233019','GSM1233020','GSM1233021','GSM1233022','GSM1233023','GSM1233024','GSM1233025','GSM1233026','GSM1233027','GSM1233028','GSM1233029','GSM1233030','GSM1233031','GSM1233032','GSM1233033','GSM1233034','GSM1233035','GSM1233036','GSM1233037','GSM1233038','GSM1233039','GSM1233040','GSM1233041','GSM1233042','GSM1233043','GSM1233044','GSM1233012','GSM1233046','GSM1233047','GSM1233048','GSM1233049','GSM1233050','GSM1233051','GSM1233052','GSM1233053','GSM1233054','GSM1233055','GSM1233056','GSM1233057','GSM1233058','GSM1233059','GSM1233060','GSM1233061','GSM1233062','GSM1233063','GSM1233064','GSM1233065','GSM1233066','GSM1233067','GSM1233068','GSM1233069','GSM1233070','GSM1233071','GSM1233072','GSM1233073','GSM1233074','GSM1233075','GSM1233076','GSM1233077','GSM1233078','GSM1233079','GSM1233080','GSM1233081','GSM1233082','GSM1233083','GSM1233084','GSM1233085','GSM1233086','GSM1233087','GSM1233088','GSM1233089','GSM1233090','GSM1233091','GSM1233092','GSM1233093','GSM1233094','GSM1233095','GSM1233096','GSM1233097','GSM1233098','GSM1233099','GSM1233100','GSM1233101','GSM1233102','GSM1233103','GSM1233104','GSM1233105','GSM1233106','GSM1233107','GSM1233108','GSM1233109','GSM1233110','GSM1233111','GSM1233112','GSM1233113','GSM1233114','GSM1233115','GSM1233116','GSM1233117','GSM1233118','GSM1233119','GSM1233120','GSM1233121','GSM1233122','GSM1233123','GSM1233124','GSM1233125','GSM1233126','GSM1233127','GSM1233128','GSM1233129','GSM1233130','GSM1233131','GSM1233132','GSM1233133','GSM1233134','GSM1233135','GSM1233136','GSM1233137','GSM1233138','GSM1233139','GSM1233140']
def classification_model(model,data,prediction_input,output):
    model.fit(data[prediction_input],data[output])
    predictions = model.predict(data[prediction_input])
    accuracy = metrics.accuracy_score(predictions,data[output])
    print("Accuracy : %s" % "{0:.3%}".format(accuracy))
model = DecisionTreeClassifier()
prediction_var = ['GSM1232992','GSM1232993','GSM1232994','GSM1232995','GSM1232996','GSM1232997','GSM1232998','GSM1232999','GSM1233000','GSM1233001','GSM1233002','GSM1233003','GSM1233004','GSM1233005','GSM1233006','GSM1233007','GSM1233008','GSM1233009','GSM1233010','GSM1233011','GSM1233012','GSM1233013','GSM1233014','GSM1233015','GSM1233016','GSM1233017','GSM1233018','GSM1233019','GSM1233020','GSM1233021','GSM1233022','GSM1233023','GSM1233024','GSM1233025','GSM1233026','GSM1233027','GSM1233028','GSM1233029','GSM1233030','GSM1233031','GSM1233032','GSM1233033','GSM1233034','GSM1233035','GSM1233036','GSM1233037','GSM1233038','GSM1233039','GSM1233040','GSM1233041','GSM1233042','GSM1233043','GSM1233044','GSM1233012','GSM1233046','GSM1233047','GSM1233048','GSM1233049','GSM1233050','GSM1233051','GSM1233052','GSM1233053','GSM1233054','GSM1233055','GSM1233056','GSM1233057','GSM1233058','GSM1233059','GSM1233060','GSM1233061','GSM1233062','GSM1233063','GSM1233064','GSM1233065','GSM1233066','GSM1233067','GSM1233068','GSM1233069','GSM1233070','GSM1233071','GSM1233072','GSM1233073','GSM1233074','GSM1233075','GSM1233076','GSM1233077','GSM1233078','GSM1233079','GSM1233080','GSM1233081','GSM1233082','GSM1233083','GSM1233084','GSM1233085','GSM1233086','GSM1233087','GSM1233088','GSM1233089','GSM1233090','GSM1233091','GSM1233092','GSM1233093','GSM1233094','GSM1233095','GSM1233096','GSM1233097','GSM1233098','GSM1233099','GSM1233100','GSM1233101','GSM1233102','GSM1233103','GSM1233104','GSM1233105','GSM1233106','GSM1233107','GSM1233108','GSM1233109','GSM1233110','GSM1233111','GSM1233112','GSM1233113','GSM1233114','GSM1233115','GSM1233116','GSM1233117','GSM1233118','GSM1233119','GSM1233120','GSM1233121','GSM1233122','GSM1233123','GSM1233124','GSM1233125','GSM1233126','GSM1233127','GSM1233128','GSM1233129','GSM1233130','GSM1233131','GSM1233132','GSM1233133','GSM1233134','GSM1233135','GSM1233136','GSM1233137','GSM1233138','GSM1233139','GSM1233140']
outcome_var= "Type"
classification_model(model,data,prediction_var,outcome_var)
Accuracy : 100.000%
model = svm.SVC()
classification_model(model,data,prediction_var,outcome_var)
/home/student/.local/lib/python3.5/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
Accuracy : 66.381%
model = KNeighborsClassifier()
classification_model(model,data,prediction_var,outcome_var)
Accuracy : 71.907%
model = RandomForestClassifier(n_estimators=100)
classification_model(model,data,prediction_var,outcome_var)
Accuracy : 100.000%
model=LogisticRegression()
classification_model(model,data,prediction_var,outcome_var)
/home/student/.local/lib/python3.5/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
/home/student/.local/lib/python3.5/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
  "this warning.", FutureWarning)
Accuracy : 66.041%
model=LinearDiscriminantAnalysis()
classification_model(model,data,prediction_var,outcome_var)
Accuracy : 66.041%
/home/student/.local/lib/python3.5/site-packages/sklearn/discriminant_analysis.py:388: UserWarning: Variables are collinear.
  warnings.warn("Variables are collinear.")
 
data_X= data[prediction_var]
data_y= data["Type"]
def Classification_model_gridsearchCV(model,param_grid,data_X,data_y):
    clf = GridSearchCV(model,param_grid,cv=10,scoring="accuracy")
    clf.fit(train_X,train_y)
    print("The best parameter found on development set is :")
    print(clf.best_params_)
    print("the bset estimator is ")
    print(clf.best_estimator_)
    print("The best score is ")
    print(clf.best_score_)
param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
              'min_samples_split': [2,3,4,5,6,7,8,9,10], 
              'min_samples_leaf':[2,3,4,5,6,7,8,9,10] }
model= DecisionTreeClassifier()
Classification_model_gridsearchCV(model,param_grid,data_X,data_y)
The best parameter found on development set is :
{'max_features': 'sqrt', 'min_samples_split': 5, 'min_samples_leaf': 8}
the bset estimator is 
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features='sqrt', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=8, min_samples_split=5,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')
The best score is 
0.5523620401337793
model = KNeighborsClassifier()
k_range = list(range(1, 30))
leaf_size = list(range(1,30))
weight_options = ['Type']
param_grid = {'n_neighbors': k_range, 'leaf_size': leaf_size, 'weights': weight_options}
Classification_model_gridsearchCV(model,param_grid,data_X,data_y)
model=svm.SVC()
param_grid = [
              {'C': [1, 10, 100, 1000], 
               'kernel': ['linear']
              },
              {'C': [1, 10, 100, 1000], 
               'gamma': [0.001, 0.0001], 
               'kernel': ['rbf']
              },
 ]
Classification_model_gridsearchCV(model,param_grid,data_X,data_y)
# here we will import the libraries used for machine learning
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph. I like it most for plot
%matplotlib inline
from sklearn.linear_model import LogisticRegression # to apply the Logistic regression
from sklearn.model_selection import train_test_split # to split the data into two parts
#from sklearn.cross_validation import KFold # use for cross validation
from sklearn.model_selection import GridSearchCV# for tuning parameter
from sklearn.ensemble import RandomForestClassifier # for random forest classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC # for Support Vector Machine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics # for the check the error and accuracy of the model
# Any results you write to the current directory are saved as output.
# dont worry about the error if its not working then insteda of model_selection we can use cross_validation
data = pd.read_csv("GSE50948_series_matrix.csv",header=0)
data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 54675 entries, 0 to 54674
Columns: 157 entries, type to GSM1233147
dtypes: float64(156), object(1)
memory usage: 65.5+ MB
features_mean= list(data.columns[1:50])
features_se= list(data.columns[51:100])
features_worst=list(data.columns[101:156])
print(features_mean)
print("-----------------------------------")
print(features_se)
print("------------------------------------")
print(features_worst)
['GSM1232992', 'GSM1232993', 'GSM1232994', 'GSM1232995', 'GSM1232996', 'GSM1232997', 'GSM1232998', 'GSM1232999', 'GSM1233000', 'GSM1233001', 'GSM1233002', 'GSM1233003', 'GSM1233004', 'GSM1233005', 'GSM1233006', 'GSM1233007', 'GSM1233008', 'GSM1233009', 'GSM1233010', 'GSM1233011', 'GSM1233012', 'GSM1233013', 'GSM1233014', 'GSM1233015', 'GSM1233016', 'GSM1233017', 'GSM1233018', 'GSM1233019', 'GSM1233020', 'GSM1233021', 'GSM1233022', 'GSM1233023', 'GSM1233024', 'GSM1233025', 'GSM1233026', 'GSM1233027', 'GSM1233028', 'GSM1233029', 'GSM1233030', 'GSM1233031', 'GSM1233032', 'GSM1233033', 'GSM1233034', 'GSM1233035', 'GSM1233036', 'GSM1233037', 'GSM1233038', 'GSM1233039', 'GSM1233040']
-----------------------------------
['GSM1233042', 'GSM1233043', 'GSM1233044', 'GSM1233012.1', 'GSM1233046', 'GSM1233047', 'GSM1233048', 'GSM1233049', 'GSM1233050', 'GSM1233051', 'GSM1233052', 'GSM1233053', 'GSM1233054', 'GSM1233055', 'GSM1233056', 'GSM1233057', 'GSM1233058', 'GSM1233059', 'GSM1233060', 'GSM1233061', 'GSM1233062', 'GSM1233063', 'GSM1233064', 'GSM1233065', 'GSM1233066', 'GSM1233067', 'GSM1233068', 'GSM1233069', 'GSM1233070', 'GSM1233071', 'GSM1233072', 'GSM1233073', 'GSM1233074', 'GSM1233075', 'GSM1233076', 'GSM1233077', 'GSM1233078', 'GSM1233079', 'GSM1233080', 'GSM1233081', 'GSM1233082', 'GSM1233083', 'GSM1233084', 'GSM1233085', 'GSM1233086', 'GSM1233087', 'GSM1233088', 'GSM1233089', 'GSM1233090']
------------------------------------
['GSM1233092', 'GSM1233093', 'GSM1233094', 'GSM1233095', 'GSM1233096', 'GSM1233097', 'GSM1233098', 'GSM1233099', 'GSM1233100', 'GSM1233101', 'GSM1233102', 'GSM1233103', 'GSM1233104', 'GSM1233105', 'GSM1233106', 'GSM1233107', 'GSM1233108', 'GSM1233109', 'GSM1233110', 'GSM1233111', 'GSM1233112', 'GSM1233113', 'GSM1233114', 'GSM1233115', 'GSM1233116', 'GSM1233117', 'GSM1233118', 'GSM1233119', 'GSM1233120', 'GSM1233121', 'GSM1233122', 'GSM1233123', 'GSM1233124', 'GSM1233125', 'GSM1233126', 'GSM1233127', 'GSM1233128', 'GSM1233129', 'GSM1233130', 'GSM1233131', 'GSM1233132', 'GSM1233133', 'GSM1233134', 'GSM1233135', 'GSM1233136', 'GSM1233137', 'GSM1233138', 'GSM1233139', 'GSM1233140', 'GSM1233141', 'GSM1233142', 'GSM1233143', 'GSM1233144', 'GSM1233145', 'GSM1233146']
data['type']=data['type'].map({'M':1,'B':0})
x = data[['type','GSM1232992','GSM1232993','GSM1232994','GSM1232995','GSM1232996','GSM1232997','GSM1232998','GSM1232999','GSM1233000','GSM1233001','GSM1233002','GSM1233003','GSM1233004','GSM1233005','GSM1233006','GSM1233007','GSM1233008','GSM1233009','GSM1233010','GSM1233011','GSM1233012','GSM1233013','GSM1233014','GSM1233015','GSM1233016','GSM1233017','GSM1233018','GSM1233019','GSM1233020','GSM1233021','GSM1233022','GSM1233023','GSM1233024','GSM1233025','GSM1233026','GSM1233027','GSM1233028','GSM1233029','GSM1233030','GSM1233031','GSM1233032','GSM1233033','GSM1233034','GSM1233035','GSM1233036','GSM1233037','GSM1233038','GSM1233039','GSM1233040','GSM1233041','GSM1233042','GSM1233043','GSM1233044','GSM1233012','GSM1233046','GSM1233047','GSM1233048','GSM1233049','GSM1233050','GSM1233051','GSM1233052','GSM1233053','GSM1233054','GSM1233055','GSM1233056','GSM1233057','GSM1233058','GSM1233059','GSM1233060','GSM1233061','GSM1233062','GSM1233063','GSM1233064','GSM1233065','GSM1233066','GSM1233067','GSM1233068','GSM1233069','GSM1233070','GSM1233071','GSM1233072','GSM1233073','GSM1233074','GSM1233075','GSM1233076','GSM1233077','GSM1233078','GSM1233079','GSM1233080','GSM1233081','GSM1233082','GSM1233083','GSM1233084','GSM1233085','GSM1233086','GSM1233087','GSM1233088','GSM1233089','GSM1233090','GSM1233091','GSM1233092','GSM1233093','GSM1233094','GSM1233095','GSM1233096','GSM1233097','GSM1233098','GSM1233099','GSM1233100','GSM1233101','GSM1233102','GSM1233103','GSM1233104','GSM1233105','GSM1233106','GSM1233107','GSM1233108','GSM1233109','GSM1233110','GSM1233111','GSM1233112','GSM1233113','GSM1233114','GSM1233115','GSM1233116','GSM1233117','GSM1233118','GSM1233119','GSM1233120','GSM1233121','GSM1233122','GSM1233123','GSM1233124','GSM1233125','GSM1233126','GSM1233127','GSM1233128','GSM1233129','GSM1233130','GSM1233131','GSM1233132','GSM1233133','GSM1233134','GSM1233135','GSM1233136','GSM1233137','GSM1233138','GSM1233139','GSM1233140','GSM1233141','GSM1233142','GSM1233143','GSM1233144','GSM1233145','GSM1233146','GSM1233147']
]
y = data.type
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.30,random_state=42)
model = []
model.append(('LR',LogisticRegression()))
model.append(('LDA',LinearDiscriminantAnalysis()))
model.append(('KNN',KNeighborsClassifier()))
model.append(('CART',DecisionTreeClassifier()))
model.append(('NB',GaussianNB()))
model.append(('svm',SVC()))
model.append(('RF',RandomForestClassifier()))
result = []
names = []
from sklearn import model_selection
for name,models in model:
    kfold = model_selection.KFold(n_splits=10,random_state=7)
    CV_result = model_selection.cross_val_score(models,x_train,y_train,cv=kfold,scoring='accuracy')
    result.append(CV_result)
    names.append(name)
    msg = '%s,%f(%f)'%(name,CV_result.mean(),CV_result.std())
    print(msg)
/home/student/.local/lib/python3.5/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
/home/student/.local/lib/python3.5/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
/home/student/.local/lib/python3.5/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
/home/student/.local/lib/python3.5/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
/home/student/.local/lib/python3.5/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
/home/student/.local/lib/python3.5/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
/home/student/.local/lib/python3.5/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
/home/student/.local/lib/python3.5/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
/home/student/.local/lib/python3.5/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
/home/student/.local/lib/python3.5/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
LR,1.000000(0.000000)
/home/student/.local/lib/python3.5/site-packages/sklearn/discriminant_analysis.py:388: UserWarning: Variables are collinear.
  warnings.warn("Variables are collinear.")
/home/student/.local/lib/python3.5/site-packages/sklearn/discriminant_analysis.py:388: UserWarning: Variables are collinear.
  warnings.warn("Variables are collinear.")
/home/student/.local/lib/python3.5/site-packages/sklearn/discriminant_analysis.py:388: UserWarning: Variables are collinear.
  warnings.warn("Variables are collinear.")
/home/student/.local/lib/python3.5/site-packages/sklearn/discriminant_analysis.py:388: UserWarning: Variables are collinear.
  warnings.warn("Variables are collinear.")
/home/student/.local/lib/python3.5/site-packages/sklearn/discriminant_analysis.py:388: UserWarning: Variables are collinear.
  warnings.warn("Variables are collinear.")
/home/student/.local/lib/python3.5/site-packages/sklearn/discriminant_analysis.py:388: UserWarning: Variables are collinear.
  warnings.warn("Variables are collinear.")
/home/student/.local/lib/python3.5/site-packages/sklearn/discriminant_analysis.py:388: UserWarning: Variables are collinear.
  warnings.warn("Variables are collinear.")
/home/student/.local/lib/python3.5/site-packages/sklearn/discriminant_analysis.py:388: UserWarning: Variables are collinear.
  warnings.warn("Variables are collinear.")
/home/student/.local/lib/python3.5/site-packages/sklearn/discriminant_analysis.py:388: UserWarning: Variables are collinear.
  warnings.warn("Variables are collinear.")
/home/student/.local/lib/python3.5/site-packages/sklearn/discriminant_analysis.py:388: UserWarning: Variables are collinear.
  warnings.warn("Variables are collinear.")
LDA,0.624320(0.005667)
KNN,0.818666(0.004752)
CART,1.000000(0.000000)
NB,1.000000(0.000000)
/home/student/.local/lib/python3.5/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
/home/student/.local/lib/python3.5/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
/home/student/.local/lib/python3.5/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
/home/student/.local/lib/python3.5/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
/home/student/.local/lib/python3.5/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
/home/student/.local/lib/python3.5/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
/home/student/.local/lib/python3.5/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
/home/student/.local/lib/python3.5/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
/home/student/.local/lib/python3.5/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
/home/student/.local/lib/python3.5/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
svm,0.996081(0.000851)
/home/student/.local/lib/python3.5/site-packages/sklearn/ensemble/forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
/home/student/.local/lib/python3.5/site-packages/sklearn/ensemble/forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
/home/student/.local/lib/python3.5/site-packages/sklearn/ensemble/forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
/home/student/.local/lib/python3.5/site-packages/sklearn/ensemble/forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
/home/student/.local/lib/python3.5/site-packages/sklearn/ensemble/forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
/home/student/.local/lib/python3.5/site-packages/sklearn/ensemble/forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
/home/student/.local/lib/python3.5/site-packages/sklearn/ensemble/forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
/home/student/.local/lib/python3.5/site-packages/sklearn/ensemble/forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
/home/student/.local/lib/python3.5/site-packages/sklearn/ensemble/forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
/home/student/.local/lib/python3.5/site-packages/sklearn/ensemble/forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
RF,0.998903(0.001138)
Result
(1)Linear Regression accuracy(LR,1.000000(0.000000) (2)Logistic Regression accuracy(LDA,0.624320(0.005667)) (3)K-nearest neighbors accuracy(KNN,0.818666(0.004752)) (4)Decision Tree accuracy(CART,1.000000(0.000000)) (5)Naive Bayes accuracy(NB,1.000000(0.000000)) (5)SVC(svm,0.996081(0.000851) (5)RandomForestClassifier(RF,0.998903(0.001138))
 
