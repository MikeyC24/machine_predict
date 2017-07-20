"""
what does a user need to input
1. classification v prediction amount
2. set model params v optimizer from grid search
3. which to use, right now logisitc, tree, nnl
4. how to run on data simple, train, kfold
5. what it gives back after iterating over varibles
	a. tpr, fpr
	b. error score for each chose model

this can be a dictionary
1. type 1 or 0
2. constant params v optmizing (save this for later much more to be done)
3. dict keys [logistic, decision_tree, neural_network]
4. simple, train, kfold
5. dict values : error metric and score, tpr, fpr

model:{error_metric:mse, error_sig_level:.05, tpr_range[.6,.8]}