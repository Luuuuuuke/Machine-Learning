import os
import csv
import ml
import BaysianTraining
import gender
import NN_ope
import NN_agr
import LIWCBig5
import oxford_KNNGender
import oxford_KNNAge

def readprofile(testdata_file_path, output_file_path):
    
    #Machine Learning algorithm to get the prediction results
  
    #base line
    model = ml.training("/data/training/profile/profile.csv")  #training data
    m2 = str(round(model[2],2))
    m3 = str(round(model[3],2))
    m4 = str(round(model[4],2))
    m5 = str(round(model[5],2))
    m6 = str(round(model[6],2))
    
    #KNN for gender by oxford.csv
    print 'starting KNN learning for gender prediction by oxford.csv'
    OxfordKNNGenderModel = oxford_KNNGender.train()
    print 'KNN learning end.'

    #KNN for age by oxford.csv
    print 'starting KNN learning for age prediction by oxford.csv'
    OxfordKNNAgeModel = oxford_KNNAge.train()
    print 'KNN learning end.'

    #Baysian for ang and gender
    print 'starting Baysian training for age and gender...'
    baysianModel = BaysianTraining.baysianTraining("/data/training/profile/profile.csv")
    print 'Baysian training model end.'
	
    #Neural Network for gender
    print 'starting Neural Network learning for gender prediction...'
    NNGenderModel = gender.neuralNetworkTraining_gender()
    print 'Neural Network learning end.'
    
    #Neural Network for ope
    #print 'starting Neural Network learning for ope prediction...'
    #NNOpeModel = NN_ope.neuralNetworkTraining_ope()
    #print 'Neural Network learning end.'
    
    #Neural Network for agr
    #print 'starting Neural Network learning for agr prediction...'
    #NNAgrModel = NN_agr.neuralNetworkTraining_agr()
    #print 'Neural Network learning end.'
    
    #LIWC linear regression training
    print 'start linear regression training with LIWC on big5: '
    LIWC_model = LIWCBig5.train()
    print 'Linear regression training with LIWC on big5 done. '

    #output the results
    userID = "";
    #if the oupuht file path does not exit, build it
    if(os.path.isdir(output_file_path) == False):
        os.makedirs(output_file_path)
        pass
    #open profile.csv from testing data folder to get every user id
    filepath = testdata_file_path + "/profile/profile.csv"
    
    with open(filepath,"r") as file:                 
        profile_dict = csv.DictReader(file)
        for row in profile_dict:
            print '==================================================='
            userID = row['userid']
            print 'start processing userId: ',  userID

            #KNN by oxford.csv for gender if there is a record
            print 'start prediction on gender with oxford.csv... '
            oxford_KNN_gender = oxford_KNNGender.classify(OxfordKNNGenderModel, testdata_file_path, userID)
            print 'gender predicted by oxford.csv with KNN: ', oxford_KNN_gender

	    #KNN by oxford.csv for age if there is a record
            print 'start prediction on gender with oxford.csv... '
            oxford_KNN_age = oxford_KNNAge.classify(OxfordKNNAgeModel, testdata_file_path, userID)
            print 'age predicted by oxford.csv with KNN: ', oxford_KNN_age

            #LIWC Linear Regression prediction for big5
            print 'start linear regression prediction on big 5 with LIWC... '
            liwcbig5 = LIWCBig5.classify(LIWC_model[0], LIWC_model[1], LIWC_model[2], LIWC_model[3], LIWC_model[4], testdata_file_path, userID);
            print 'linear regression prediction with LIWC on big5:', liwcbig5

            #Baysian for age and gender
            print 'Bayesian prediction starting...'
	    txt_file_path = testdata_file_path + "/text/" + userID + ".txt"
            newText = open(txt_file_path).read()
            print 'readed text file: ', txt_file_path
            print 'Start processing ', txt_file_path, 'by Bayesian trained model...'
	    results = BaysianTraining.classifier(baysianModel,newText)
            print 'result gotten by Bayesian: ', results[0], ', ', results[1]

            #Neural Network for gender
            image_file_path = testdata_file_path + "/image/" + userID + ".jpg"
            print 'Start processing ', image_file_path, 'by Neural Network trained model...'
            gender_NN = gender.classify(NNGenderModel, image_file_path)
	    print 'Gender result gotten by NN: ', gender_NN

            #ope_NN = NN_ope.classify(NNOpeModel, image_file_path)
            #print 'Ope result gotten by NN: ', ope_NN

            #agr_NN = NN_agr.classify(NNAgrModel, image_file_path)
            #print 'Agr result gotten by NN: ', agr_NN
            
            
            #we have oxford.csv, baysian to predict age, it is time to decide the final
            if oxford_KNN_age == "unknown":
                age_final = results[1] #use baysian result for age
            else:
                age_final = oxford_KNN_age  #use oxford result

            #we have oxford.csv, NN, baysian to predict gender, it is time to decide the final
            if oxford_KNN_gender == "unknown":
                gender_final = results[0] #use baysian result for gender
            else:
                gender_final = oxford_KNN_gender  #use oxford result


            #for each userID, generate a output xml file
            output_file = open(output_file_path + "/" + userID + ".xml",'w')
            output_file.write("<userId = \"{" + userID + "}\"\n"
                              "age_group = \"" + age_final + "\"\n" ##age
                              "gender = \"" + gender_final + "\"\n"  #gender
                              #"extrovert = \"" + m2 + "\"\n"        #base line
                              "extrovert = \"" + str(round(liwcbig5[1],2)) + "\"\n" #LIWC
                              #"neurotic = \"" + m3 + "\"\n"
                              "neurotic = \"" + str(round(liwcbig5[4],2)) + "\"\n" #LIWC      
                              #"agreeable = \"" + m4 + "\"\n"       #base line
                              #"agreeable = \"" + str(round(agr_NN,2)) + "\"\n"     #NN
                              "agreeable = \"" + str(round(liwcbig5[3],2)) + "\"\n" #LIWC
                              #"conscientious = \"" + m5 + "\"\n"
                              "conscientious = \"" + str(round(liwcbig5[2],2)) + "\"\n" #LIWC
                              #"open = \"" + m6 + "\"\n"            #base line
                              #"open = \"" + str(round(ope_NN,2)) + "\"\n"  #NN
                              "open = \"" + str(round(liwcbig5[0],2)) + "\"\n"  #LIWC
                              "/>")
            output_file.close()
            del output_file
            pass
    print("Done.")
    
        
