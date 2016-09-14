import os
import sys
import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.cross_validation import KFold

def work(test_userid):
    #read testing data
    df_test = pd.read_csv('/data/public-test-data/LIWC.csv',sep=',')    

    a = []          #return results
    training_liwc = '/data/training/LIWC.csv'
    training_profile = '/data/training/profile/profile.csv'

    #load data as data frame
    df_liwc = pd.read_csv(training_liwc,sep=',')
    df_profile = pd.read_csv(training_profile,sep=',')

    #join the two data frame
    df = pd.merge(left=df_liwc,right=df_profile, how='left', left_on='userId',  right_on='userid')
    
    df = df.drop('userid', 1)
    df = df.drop('Unnamed: 0', 1)
    df = df.drop('age', 1)
    df = df.drop('gender', 1)


    big5 = ['ope','ext','con','agr','neu']
    for personality in big5:
        df_personality = df
        df_personality = df_personality.drop('userId', 1)
        for trait in big5:
            df_personality = df_personality.drop(trait, 1)
              
      
        #making feature list for personality scores
        feature_list_personality = df_personality.columns.tolist()[:]

        # regression models
        X = df_personality[feature_list_personality]

        #Y = df['new_flag']
        Y = df[[personality]]

        regr = linear_model.LinearRegression()
    
        # Train the model using the training sets
        regr.fit(X, Y)
    
        #get input
        user_x = df_test.loc[df_test['userId'] == test_userid]
        user_x = user_x.drop('userId', 1)  
        #print 'get test user input: ', user_x       
        #predict
        result  = regr.predict(user_x)
        a.append(result[0])

    return a

def train():
    #read testing data
    df_test = pd.read_csv('/data/public-test-data/LIWC.csv',sep=',')    

    training_liwc = '/data/training/LIWC.csv'
    training_profile = '/data/training/profile/profile.csv'

    #load data as data frame
    df_liwc = pd.read_csv(training_liwc,sep=',')
    df_profile = pd.read_csv(training_profile,sep=',')

    #join the two data frame
    df = pd.merge(left=df_liwc,right=df_profile, how='left', left_on='userId',  right_on='userid')
    
    df = df.drop('userid', 1)
    df = df.drop('Unnamed: 0', 1)
    df = df.drop('age', 1)
    df = df.drop('gender', 1)


    big5 = ['ope','ext','con','agr','neu']
    
    df_personality = df
    df_personality = df_personality.drop('userId', 1)

    for trait in big5:
        df_personality = df_personality.drop(trait, 1)
              
      
    #making feature list for personality scores
    feature_list_personality = df_personality.columns.tolist()[:]

    # features
    X = df_personality[feature_list_personality]
    X = X.as_matrix()

    #Y is labels
    Y_ope = df['ope']
    Y_ext = df['ext']
    Y_con = df['con']
    Y_agr = df['agr']
    Y_neu = df['neu']

    regr_ope = linear_model.LinearRegression()
    regr_ext = linear_model.LinearRegression()
    regr_con = linear_model.LinearRegression()
    regr_agr = linear_model.LinearRegression()
    regr_neu = linear_model.LinearRegression()
    
    #10-fold 
    kf = KFold(9500, n_folds=10)
    mylist = list(kf)
    train, test = mylist[8]
    Xope_train, Xope_test, yope_train, yope_test = X[train], X[test], Y_ope[train], Y_ope[test]
    Xext_train, Xext_test, yext_train, yext_test = X[train], X[test], Y_ext[train], Y_ext[test]
    Xcon_train, Xcon_test, ycon_train, ycon_test = X[train], X[test], Y_con[train], Y_con[test]
    Xagr_train, Xagr_test, yagr_train, yagr_test = X[train], X[test], Y_agr[train], Y_agr[test]
    Xneu_train, Xneu_test, yneu_train, yneu_test = X[train], X[test], Y_neu[train], Y_neu[test]

    # Train the model using the training sets
    regr_ope.fit(Xope_train, yope_train)
    regr_ext.fit(Xext_train, yext_train)
    regr_con.fit(Xcon_train, ycon_train)
    regr_agr.fit(Xagr_train, yagr_train)
    regr_neu.fit(Xneu_train, yneu_train)
    

    return [regr_ope, regr_ext, regr_con, regr_agr, regr_neu]

def classify(regr_ope, regr_ext, regr_con, regr_agr, regr_neu, testdata_file_path, test_userid):
    tempfilefolder = os.getenv("HOME")+"//tempfile"
    if(os.path.isdir(tempfilefolder) == False):
        os.makedirs(tempfilefolder)
        pass

    inputLIWCFile = testdata_file_path + '/LIWC.csv'

    #read testing data
    df_test = pd.read_csv(inputLIWCFile, sep=',')
    print 'LIWC features readed from: ',  inputLIWCFile
    

    #get input
    user_x = df_test.loc[df_test['userId'] == test_userid]
    if user_x.empty:
        error_msg = 'can not look up features by the userid:' + test_userid
        #print 'LIWC error, there are no features found by the userid:', test_userid
        sys.stderr.write(error_msg)
        sys.exit(1)
 
    user_x = user_x.drop('userId', 1)  
        
    #predict
    ope = regr_ope.predict(user_x)
    ext = regr_ext.predict(user_x)
    con = regr_con.predict(user_x)
    agr = regr_agr.predict(user_x)
    neu = regr_neu.predict(user_x)
	
	
    return [ope, ext, con, agr, neu]

