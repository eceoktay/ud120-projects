#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here

    
    cleaned_data = [(ages[index][0], net_worths[index][0], abs(predictions[index][0] - net_worths[index][0])) for index in range(len(predictions))] 
    cleaned_data.sort(key=lambda tup: tup[2])
    
    return cleaned_data[0: int(len(cleaned_data) * 0.9)]

