#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

print "Number of data points (people): ", len(enron_data)

print "Number of features for each person: ", len(enron_data["METTS MARK"])

number_of_poi = 0
for key in enron_data:
    if enron_data[key]["poi"] == 1:
        number_of_poi = number_of_poi + 1
print "Number of POIs: ", number_of_poi

print "Total value of the stock belonging to James Prentice: ", enron_data["PRENTICE JAMES"]["total_stock_value"]

print "Number of email messages from Wesley Colwell to POI: ", enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]

print "Value of stock options exercised by Jeffrey K Skilling: ", enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

individuals = ["Lay", "Skilling", "Fastow"]
person = "Lay"
value = 0
for key in enron_data:
    for i in range(len(individuals)):
        if individuals[i].lower() in key.lower():
            if enron_data[key]["total_payments"] >= value:
                value = enron_data[key]["total_payments"]
                person = individuals[i]
print "Of these three individuals (Lay, Skilling and Fastow);", person, "took home the most money"
print "That person get: ", value

number_of_quantified_salaries = 0
number_of_known_emails = 0
for key in enron_data:
    #print "Salary(", key, "): ", enron_data[key]["salary"]
    if (enron_data[key]["salary"] != 'NaN'):
        number_of_quantified_salaries = number_of_quantified_salaries + 1
    #print "Email(", key, "): ", enron_data[key]["email_address"]
    if (enron_data[key]["email_address"] != 'NaN'):
        number_of_known_emails = number_of_known_emails + 1
print "Number of folks in this dataset have a quantified salary: ", number_of_quantified_salaries
print "Number of folks in this dataset have a known salary: ", number_of_known_emails

number_of_nan_total_payments = 0
for key in enron_data:
    if (enron_data[key]["total_payments"] == 'NaN'):
        number_of_nan_total_payments = number_of_nan_total_payments + 1
print "Number of folks in this dataset have a known salary: ", number_of_nan_total_payments
print "Percentage of people: ", number_of_nan_total_payments * 100 / len(enron_data), "%"

number_of_poi_nan_total_payments = 0
for key in enron_data:
    if (enron_data[key]["poi"] == 1) and (enron_data[key]["total_payments"] == 'NaN'):
        number_of_poi_nan_total_payments = number_of_poi_nan_total_payments + 1
print "Number of POIs in this dataset have a known salary: ", number_of_poi_nan_total_payments
print "Percentage of people: ", number_of_poi_nan_total_payments * 100 / len(enron_data), "%"