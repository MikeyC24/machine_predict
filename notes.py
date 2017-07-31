#notes taken from various pages to refractor

"""
#thoughts 
1. everything above is for classifers
2. could models be mixed and match for different decisions, such as decision tree to predict when right
and nnl to weight when wrong. 
3. need to set up to iterate over multi time periods on data and cycle thru vars 
"""

"""
how this should be
1. put in various vars on top
2. pick models to run
3. pick tpr and fpr ranges, with error compared against some
t value and return if statistically sifnifcant or not
4. print out data, which model, which params, which vars, and score (error, tpr, fpr)
5. run that on new data not seen (time period ahead)
"""

"""
whats next.....
in this order
1. refactor so everything is taken in at start of class
2. refactor all regressions to out fit, and out put regressions
3. have the error scores be part of this MachinePredictModel class and return everything as dict
4. set up method to iterate over various varaiables
5. check if chose error matetric is stat significant
6. if stat significant return return scores if they hit a certain range
"""

"""
7/24/17
basically working in terms of user inputs v outputs
next....
1. determine stat significance of vars?
2. how to have data constantly feed in this
3. take all of that and run on new data 
4. return comvo variables sets by some type of order 
"""

"""
idea to take in user variables, have user input variables, then for all of them 
have method to turn in array?dict? and input from there? 
"""
"""
major milestones next - 
AA. when tests are done, fix this method _set_up_data_for_models_test(
all that needs to be done is to add in a *arg where defaults is self.columns_all
# or can take in a custom set are when the need to cycle thru the vars arrives
1.have the var returns be more readable than
the sloppy way the key is returned now
2. be able to store all answers in database, this may  mean making
some of th return dicts more aligned
3. be able to get data out of those databases to run future tests
"""