"""
break down of this class
this class will call on all the others to.....
1. clean data for use, 2. create and build new coulms, dfs, etc 
3. the data will be fed to each model, each model will cycle thru data/vars
until it returns a binary yes or no prediction based on optmized/acceptable
error scores 4. all these scores (and error metrics) are then fed to predictmodel
where a majority voting vote on them leading to a final yes or no
INPUTs for this class 1. variable to predict, models to use and their allowed scores


first 3 models
1. decision tree
2. bayes
3 ? 
