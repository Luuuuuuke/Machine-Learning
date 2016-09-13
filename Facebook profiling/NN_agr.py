#neural network predicting agr
import sys
import csv
import cv2
import math
import random

#neuronet work configuration
LEARNING_RATE = 0.2
N_INPUT = 900
N_HIDDEN = 3
N_OUTPUT = 1
MOMENTUM = 0.3


def neuralNetworkTraining_agr():
    global LEARNING_RATE
    #initialize all weight wi with random small value range from -0.5 to 0.5
    #weight from input units to hidden units
    w_1 = [[0 for col in range(N_INPUT)] for row in range(N_HIDDEN)]
    delta_w_1 = [[0 for col in range(N_INPUT)] for row in range(N_HIDDEN)]
    for j in range(0,N_HIDDEN,1):  #hidden layer
        for i in range(0,N_INPUT,1):   #input layer        
            w_1[j][i] = random.random()*2-1 # w[j][i] means the weight from unit i to unit j
            delta_w_1[j][i] = 0

    #weight from hidden units to output unit
    w_2 = [0 for col in range(N_HIDDEN)]
    delta_w_2 = [0 for col in range(N_HIDDEN)]
    for i in range(0,N_HIDDEN):
        w_2[i] = random.random()*2-1
        delta_w_2[i] = 0

    face_cascade = cv2.CascadeClassifier('/Home/itadmin/haarcascade_frontalface_default.xml')

    #start 
    trainingdata_path_root = "/data/training/"
    trainingimage_path_root = trainingdata_path_root + "image/"
    traininglabels_path = trainingdata_path_root + "profile/profile.csv"

    #read csv file and store the <userid, label> dictionaries
    dict_id_agr = {}
    with open(traininglabels_path) as file:
        file_dict = csv.DictReader(file)
        count = 0
        for row in file_dict:
            dict_id_agr[row['userid']] = float(row['agr'])
            count += 1
            pass

    #for each traning example
    flag = 0
    for key,val in dict_id_agr.items():
        userid = key
        agr = val
        #find the image through userid
        each_image_path = trainingimage_path_root + userid + ".jpg"
        each_image = cv2.imread(each_image_path)
        gray = cv2.cvtColor(each_image, cv2.COLOR_BGR2GRAY)
        #face detection, find all the faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        #if no faces read, treat whole image as face
        if(len(faces) <= 0):
            crop_img = each_image
        else:
            #only consider the biggest face in the image
            biggest_face_size = 0
            for (x,y,w,h) in faces:
                #pick the main face with biggest w*h and get the face image
                if(w * h > biggest_face_size):
                    crop_img = each_image[y:y+h,x:x+w]
                    biggest_face_size = w * h;
        #unify all the face image to the 20*20 pixels
        resized_face = cv2.resize(crop_img,(30,30))
        resized_face = cv2.resize(each_image,(30,30))
        #change the image mode from RGB to intensity
        ready_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)

        #transform the face image to pixels
        input_pixels = []
        for i in range(0,30,1):
            for j in range(0, 30, 1):
                #scale intensity to make input range from 0 to 1
                scaled_intensity = float(ready_face[i][j]) / 255
                input_pixels.append(scaled_intensity)

        target_agr = float(val) / 5
        #-----------------for weight from hidden layer layer to output layer--------------
        net_hidden = [0 for col in range(N_HIDDEN)]
        output_hidden = [0 for col in range(N_HIDDEN)]
        #compute the output of all the hidden unit first
        for j in range(0,N_HIDDEN,1):
            for i in range(0, N_INPUT,1):
                net_hidden[j] += input_pixels[i] * w_1[j][i];

        for j in range(0, N_HIDDEN, 1):
            output_hidden[j] = 1.0 / (1 + pow(math.e, (-net_hidden[j])))

        #compute the output 
        net_output = 0
        for j in range(0, N_HIDDEN, 1):
            net_output += w_2[j] * output_hidden[j]
        output = 1.0 / (1 + pow(math.e, (-net_output)))
        #compute the delta
        delta_w_2 = [0 for col in range(N_HIDDEN)]
        for j in range(0, N_HIDDEN, 1):
            delta_w_2[j] = LEARNING_RATE * (target_agr - output) * output * (1 - output) * output_hidden[j] + MOMENTUM * delta_w_2[j]
        #update the weight
        pre_w_2 = [0 for col in range(N_HIDDEN)]
        for j in range(0,N_HIDDEN,1):
            pre_w_2[j] = w_2[j]
            w_2[j] += delta_w_2[j]
        #-----------------ends for weight from hidden layer layer to output layer--------------

        #-----------------for weight from input layer layer to hidden layer--------------
        #compute error term for hidden units
        error_term_output = output * (1 - output) * (target_agr - output)
        error_term_hidden = [0 for col in range(N_HIDDEN)]
        for j in range(0,N_HIDDEN,1):
            error_term_hidden[j] = output_hidden[j] * (1 - output_hidden[j]) * error_term_output * pre_w_2[j]
        #compute delta weight and update weight
        for j in range(0,N_HIDDEN,1):
            for i in range(0, N_INPUT, 1):
                delta_w_1[j][i] = LEARNING_RATE * error_term_hidden[j] * input_pixels[i] + MOMENTUM * delta_w_1[j][i]
                w_1[j][i] += delta_w_1[j][i]
        pass
        #LEARNING_RATE = LEARNING_RATE * 3 / 4
    return [w_1, w_2]

def classify(trainedModel,image_path):
    
    w_1 = trainedModel[0]
    w_2 = trainedModel[1]
    

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #face detection, find all the faces
    face_cascade = cv2.CascadeClassifier('/Home/itadmin/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if(len(faces) <= 0):
        crop_img = img
    else:
        #only consider the biggest face in the image
        biggest_face_size = 0
        for (x,y,w,h) in faces:
            #pick the main face with biggest w*h and get the face image
            if(w * h > biggest_face_size):
                crop_img = img[y:y+h,x:x+w]
                biggest_face_size = w * h;
    #unify all the face image to the 20*20 pixels
    resized_face = cv2.resize(crop_img,(30,30))
    #change the image mode from RGB to intensity
    ready_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)

    #transform the face image to pixels
    input_pixels = []
    for i in range(0,30,1):
        for j in range(0, 30, 1):
            #scale intensity to make input range from 0 to 1
            scaled_intensity = float(ready_face[i][j]) / 255
            input_pixels.append(scaled_intensity)
    
    #input the new instance into the neural network
    #compute the outputs of hidden layer
    output_hidden = [0 for col in range(N_HIDDEN)]
    net_hidden = [0 for col in range(N_HIDDEN)]
    for j in range(0, N_HIDDEN, 1):
        for i in range(0, N_INPUT, 1):
            net_hidden[j] += input_pixels[i] * w_1[j][i]
    #print(net_hidden)
    for j in range(0, N_HIDDEN, 1):
        output_hidden[j] = 1.0 / (1 + pow(math.e, (-0.1*net_hidden[j])))
    #print(output_hidden)
    #compute the final output
    output = 0
    net_output = 0
    for j in range(0,N_HIDDEN,1):
        net_output += output_hidden[j] * w_2[j]

    output = 1.0 / (1 + pow(math.e, (-net_output))) * 5
    #print(w_2)
    #print(net_output)
    #print(output)

    return output
