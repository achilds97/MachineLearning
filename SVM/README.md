For the primal SVM all of my code is in svm_subgradient. It calculates the
correct weights for the number of epochs and returns them. 

For the dual SVM the main function is svm_dual, with constraints stored in constraint and constraint2.

There are 2 getdata methods because I ended up using a different formatting for the data in the dual SVM.