import pandas as pd

from sklearn.cross_validation import KFold   
from sklearn.neighbors import KNeighborsClassifier


def train():
    training_oxford = '/data/training/oxford.csv'
    training_profile = '/data/training/profile/profile.csv'

    #load data as data frame
    df_oxford = pd.read_csv(training_oxford,sep=',')
    df_profile = pd.read_csv(training_profile,sep=',')
    
    #join the two data frame
    df = pd.merge(left=df_oxford,right=df_profile, how='left', left_on='userId',  right_on='userid')
    df = df.drop('faceID', 1)
    df = df.drop('userid', 1)
    df = df.drop('Unnamed: 0', 1)
    df = df.drop('gender', 1)
    df = df.drop('ope', 1)
    df = df.drop('con', 1)
    df = df.drop('ext', 1)
    df = df.drop('agr', 1)
    df = df.drop('neu', 1)
    
    #convert age to numeric values 0: xx-24, 1: 25-34, 2: 35-49, 3: 50-xx
    df['new_age']=0
    df.loc[df[df.age>=25].index, 'new_age'] = 1
    df.loc[df[df.age>=35].index, 'new_age'] = 2
    df.loc[df[df.age>=50].index, 'new_age'] = 3
    df = df.drop('age', 1)
    
    #get the all features X, Y
    df_features_all = df
    df_features_all = df_features_all.drop('userId', 1)  #we don't need userId as feature
    df_features_all = df_features_all.drop('new_age', 1) #we don't need this label as feature
    feature_list = df_features_all.columns.tolist()[:]
    X = df_features_all[feature_list]
    Y = df['new_age']
    
    #fold 9 is best
    kf = KFold(len(df.index), n_folds=10)
    mylist = list(kf)
    train, test = mylist[8]
    
    #train
    Xarray = X.as_matrix()
    Yarray = Y.as_matrix()
    X_train, X_test, Y_train, Y_test = Xarray[train], Xarray[test], Yarray[train], Yarray[test]
    neigh = KNeighborsClassifier(n_neighbors=5, weights='distance')
    neigh.fit(X_train, Y_train)
    
    return neigh

def classify(neigh, testdata_file_path, test_userid):
    #find oxford feature for that user
    inputOxfordFile = testdata_file_path + 'oxford.csv'
    #read testing data
    df_test = pd.read_csv(inputOxfordFile, sep=',')
    #get input
    user_x = df_test.loc[df_test['userId'] == test_userid]
    #if not found, return -1
    if user_x.empty:
        return "unknown"
    #if found, predict it
    user_x = user_x.drop('userId',1)
    user_x = user_x.drop('faceID',1)
    user_x_array = user_x.as_matrix()
    result_value = neigh.predict(user_x_array)
        
    if result_value[0] == 0.0:
        return "xx-24"
    elif result_value[0] == 1.0:
        return "25-34"
    elif result_value[0] == 2.0:
        return "35-49"
    elif result_value[0] == 3.0:
        return "50-xx" 
    else:
        return "unknown"
