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
    df = pd.merge(left=df_oxford,right=df_profile, how='left', left_on='userId', right_on='userid')
    df = df.drop('faceID', 1)
    df = df.drop('userid', 1)
    df = df.drop('Unnamed: 0', 1)
    df = df.drop('age', 1)
    df = df.drop('ope', 1)
    df = df.drop('con', 1)
    df = df.drop('ext', 1)
    df = df.drop('agr', 1)
    df = df.drop('neu', 1)
    
    #get hair features X, label Y
    df_features_hair = df
    df_features_hair = df_features_hair.drop('userId', 1)  #we don't need userId as feature
    df_features_hair = df_features_hair.drop('gender', 1) #we don't need label as feature
    feature_list = df_features_hair.columns.tolist()[:]
    for feature in feature_list:
        if not "Hair" in feature:
            df_features_hair = df_features_hair.drop(feature, 1)
    hair_feature_list = df_features_hair.columns.tolist()[:]        
    X = df_features_hair[hair_feature_list]  
    Y = df['gender']
    
    #which fold is best: k = 1
    kf = KFold(len(df.index), n_folds=10)
    mylist = list(kf)
    train, test = mylist[0]
    
    Xarray = X.as_matrix()
    X_train, X_test, Y_train, Y_test = Xarray[train], Xarray[test], Y[train], Y[test]
    neigh_hair_feature = KNeighborsClassifier(n_neighbors=5)
    neigh_hair_feature.fit(X_train, Y_train)
    
    return neigh_hair_feature
    
def classify(neigh_hair_feature, testdata_file_path, test_userid):
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
    feature_list = df_test.columns.tolist()[:]
    for feature in feature_list:
        if not "Hair" in feature:
            user_x = user_x.drop(feature, 1)
    user_x_array = user_x.as_matrix()
    result_value = neigh_hair_feature.predict(user_x_array)

    if (result_value[0] == 1.0):
        return "female"
    else:
        return "male"
