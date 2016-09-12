import csv

def training():
    predicted_age_group = ""
    predicted_gender = ""
    predicted_open = ""
    predicted_extr = ""
    predicted_neur = ""
    predicted_agre = ""
    predicted_cons = ""

    dict_age = {"18-24":0,"25-34":0,"35-49":0,"50-xx":0}
    dict_gender = {"0.0":0, "1.0":0}
    dict_extr = {}
    dict_neur = {}
    dict_agre = {}
    dict_cons = {}
    dict_open = {}

    with open('D:\\profile.csv') as file:
        file_dict = csv.DictReader(file)
        for row in file_dict:
            #set each age to dictionary <age,appear times>
            item_age = float(row['age'])
            if item_age <= 24:
                dict_age["18-24"] += 1
            elif item_age >= 25 and item_age <= 34:
                dict_age["25-34"] += 1
            elif item_age >= 35 and item_age <=49:
                dict_age["35-49"] += 1
            else:
                dict_age["50-xx"] += 1
            #set each gender to dictionary <gender,appear times>
            item_gender = row['gender']
            dict_gender[item_gender] += 1
            #set each openness to dictionary <openness, appear times>
            item_open = row["ope"]
            if item_open not in dict_open:
                dict_open.setdefault(item_open,1)
            else:
                dict_open[item_open] += 1
            #set each exrovert to dictionary <extrovert, appear times>
            item_extr = row["ext"]
            if item_extr not in dict_extr:
                dict_extr.setdefault(item_extr,1)
            else:
                dict_extr[item_extr] += 1
            #set each neurotic to dictionary <neurotic, appear times>
            item_neur = row["neu"]
            if item_neur not in dict_neur:
                dict_neur.setdefault(item_neur,1)
            else:
                dict_neur[item_neur] += 1
            #set each agreeable to dictionary <agreeable, appear times>
            item_agre = row["agr"]
            if item_agre not in dict_agre:
                dict_agre.setdefault(item_agre,1)
            else:
                dict_agre[item_agre] += 1
            #set each conscientious to dictionary <conscientious, appear times>
            item_cons = row["con"]
            if item_cons not in dict_cons:
                dict_cons.setdefault(item_cons,1)
            else:
                dict_cons[item_cons] += 1
            pass
        #find the most frequent age to be the predicted age
        max_age_times = 0
        for keys,values in dict_age.items():
            if values > max_age_times:
                max_age_times = values
                predicted_age_group = keys 
            pass
        #find the most frequent gender to be the predicted gender
        if dict_gender["0.0"] >= dict_gender["1.0"]:
            predicted_gender = "male"
        else:
            predicted_gender = "female"
        #find the most frequent openness to be predicted openness
        max_open_times = 0
        for keys,values in dict_open.items():
            if values > max_open_times:
                max_open_times = values
                predicted_open = keys
            pass
        #find the most frequent extrovert to be predicted extrovert
        max_extr_times = 0
        for keys,values in dict_extr.items():
            if values > max_extr_times:
                max_extr_times = values
                predicted_extr = keys
            pass
        #find the most frequent neurotc to be predicted neurotc
        max_neur_times = 0
        for keys,values in dict_neur.items():
            if values > max_neur_times:
                max_neur_times = values
                predicted_neur = keys
            pass
        #find the most frequent agreeable to be predicted agreeable
        max_agre_times = 0
        for keys,values in dict_agre.items():
            if values > max_agre_times:
                max_agre_times = values
                predicted_agre = keys
            pass
        #find the most frequent conscientious to be predicted conscientious
        max_cons_times = 0
        for keys,values in dict_cons.items():
            if values > max_cons_times:
                max_cons_times = values
                predicted_cons = keys
            pass
    return [predicted_age_group, predicted_gender, predicted_extr, predicted_neur, predicted_agre, predicted_cons, predicted_open]


 
