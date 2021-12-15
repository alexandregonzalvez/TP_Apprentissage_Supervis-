from sklearn.datasets import fetch_openml 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import zero_one_loss
import matplotlib.pyplot as plt 
import numpy as np
import pickle
import random
import time

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data

def splitSetRandom(data,labels, train_size):
    random_indexes = np.random.randint(labels.size, size=labels.size)
    
    X_train = np.array([data[i] for i in random_indexes[:train_size]])
    X_test = np.array([data[i] for i in random_indexes[train_size:]])
    y_train = np.array([labels[i] for i in random_indexes[:train_size]])
    y_test =  np.array([labels[i] for i in random_indexes[train_size:]])
    return X_train,X_test,y_train,y_test

# Extracting mnist
mnist = fetch_openml('mnist_784', as_frame=False) # Extraction

# Divide into training and testing
X_train,X_test,y_train,y_test = splitSetRandom(data = mnist.data, labels = mnist.target, train_size = 49000)

# Step 1
def run_simple_mlp(X_train,X_test,y_train,y_test):
    start_t = time.time()
    clf = MLPClassifier(hidden_layer_sizes = 50)
    clf.fit(X_train,y_train)
    print(f'score = {clf.score(X_test,y_test)} in {time.time()-start_t} seconds')

    # Predict image 4
    image_4 = X_test[3]
    plt.imshow(image_4.reshape(28, 28) ,cmap=plt.cm.gray_r,interpolation="nearest") 
    plt.show()
    to_predict = image_4.reshape(1, -1)
    print(f'class = {clf.predict(to_predict)[0]}')

    y_pred = clf.predict(X_test)
    precision_score(y_test,y_pred,average ='micro')

run_simple_mlp(X_train,X_test,y_train,y_test)

# Step 2
def varying_n_hidden_layer_of_1_neuron_mlp(X_train,X_test,y_train,y_test):
    results = []
    for n_hidden_layers in range(2,101,5):
        start_t = time.time()
        clf = MLPClassifier(hidden_layer_sizes = tuple([1 for i in range(n_hidden_layers)]),n_iter_no_change=4,verbose=True)
        clf.fit(X_train,y_train)
        precision = precision_score(y_test,clf.predict(X_test),average ='micro')
        duration = time.time()-start_t
        results.append({"n_hidden_layers":n_hidden_layers, "precision":precision, "time":duration})
        print(f'For {n_hidden_layers} n_hidden_layers the precision is {precision} in {duration} seconds')
    pickle.dump(results, open( "results/results_varying_hidden_layers_1_neurons.p", "wb" )) # Save results
    precisions = []
    n_hidden_layers_list = []
    for result in results:
        precisions.append(result["precision"])
        n_hidden_layers_list.append(result["n_hidden_layers"])
        
    plt.plot(n_hidden_layers_list,precisions)
    plt.xlabel('n_hidden_layers')
    plt.ylabel('precision')
    plt.title('Evolution of precision by numbers of hidden layers of 1 neuron')
    plt.show()
    return results

varying_n_hidden_layer_of_1_neuron_mlp(X_train,X_test,y_train,y_test)

# Step 3
def createRandomModel():
    hidden_layers_neurons_tuple = tuple([random.randint(10,300) for i in range(random.randint(1,10))])
    model = MLPClassifier(hidden_layer_sizes = hidden_layers_neurons_tuple,verbose=True)
    print(hidden_layers_neurons_tuple)
    return model,hidden_layers_neurons_tuple

def train(model,X_train,X_test,y_train,y_test):
    start_t = time.time()
    model.fit(X_train,y_train)
    precision = precision_score(y_test,model.predict(X_test),average ='micro')
    duration = time.time()-start_t
    print(f'The precision is {precision} in {duration} seconds')
    return precision, duration

model1,desc1 = createRandomModel()
model2,desc2 = createRandomModel()
model3,desc3 = createRandomModel()
model4,desc4 = createRandomModel()
model5,desc5 = createRandomModel()

result1,time1 = train(model1,X_train,X_test,y_train,y_test)
result2,time2 = train(model2,X_train,X_test,y_train,y_test)
result3,time3 = train(model3,X_train,X_test,y_train,y_test)
result4,time4 = train(model4,X_train,X_test,y_train,y_test)
result5,time5 = train(model5,X_train,X_test,y_train,y_test)

import math 
results = [result1,result2,result3,result4,result5]
times = [time1,time2,time3,time4,time5]
descs = [desc1,desc2,desc3,desc4,desc5]

precisions = [result for result in results]
labels =  ["{}\n{}s".format(descs[i],math.trunc(times[i]))  for i in range(5)]

f, ax = plt.subplots(figsize=(18,5)) # set the size that you'd like (width, height)
ax.legend(fontsize = 14)
plt.bar(range(5), precisions, tick_label=labels)
plt.title("precision depending on the model")
plt.ylabel("precision")
plt.xlabel("models")
plt.ylim([0.98,0.99])
plt.show()

# Step 4
def studyConvergenceDependingSolvers(X_train,X_test,y_train,y_test,hidden_layers_neurons_tuple= (91, 92, 23, 182, 87),max_iter_list = [5,10,20,50,100],solver_list = ['lbfgs','sgd','adam']):
    for solver in solver_list:
        results = []
        epoch = 0
        for max_iter in max_iter_list:
            model = MLPClassifier(hidden_layer_sizes = hidden_layers_neurons_tuple,max_iter=max_iter,solver=solver,verbose=True)   
            start_t = time.time()
            model.fit(X_train,y_train)
            training_time = time.time()-start_t
            prediction = model.predict(X_test)
            precision = precision_score(y_test,prediction,average ='micro')
            recall = recall_score(y_test,prediction,average = 'micro')
            error = zero_one_loss(y_test,prediction)
            results.append({"max_iter":max_iter,"precision":precision,"error":error,"recall":recall,"training_time":training_time})
        pickle.dump(results, open( "results/by_solver/{}_results.p".format(solver), "wb" )) # Save results
        print("finish for solver {}".format(solver))

studyConvergenceDependingSolvers(X_train,X_test,y_train,y_test,hidden_layers_neurons_tuple= (91, 92, 23, 182, 87),max_iter_list = [5,10,20,50,100],solver_list = ['lbfgs','sgd','adam'])

solvers = ['lbfgs','sgd','adam']

for solver in solvers:
    results = unpickle(f'results/by_solver/{solver}_results.p')
    precisions = [result["precision"] for result in results]
    max_iter = [result["max_iter"] for result in results]
    plt.plot(max_iter, precisions, label=solver)

plt.xlabel('total_iteration')
plt.ylabel('precision')
plt.title('Evolution of precision by iterations')
plt.legend()
plt.show()

for solver in solvers:
    results = unpickle(f'results/by_solver/{solver}_results.p')
    errors = [result["error"] for result in results]
    max_iter = [result["max_iter"] for result in results]
    plt.plot(max_iter, errors, label=solver)

plt.xlabel('total_iteration')
plt.ylabel('error')
plt.title('Evolution of error by iterations')
plt.legend()
plt.show()

for solver in solvers:
    results = unpickle(f'results/by_solver/{solver}_results.p')
    recalls = [result["recall"] for result in results]
    max_iter = [result["max_iter"] for result in results]
    plt.plot(max_iter, recalls, label=solver)

plt.xlabel('total_iteration')
plt.ylabel('recall')
plt.title('Evolution of recall by iterations')
plt.legend()
plt.show()

for solver in solvers:
    results = unpickle(f'results/by_solver/{solver}_results.p')
    precisions = [result["precision"] for result in results]
    training_time = [result["training_time"] for result in results]
    plt.plot(training_time, precisions, label=solver)

plt.xlabel('training_time')
plt.ylabel('precision')
plt.title('Evolution of precision by training_time')
plt.legend()
plt.show()

# Step 5 
def studyConvergenceDependingActivationFunction(X_train,X_test,y_train,y_test,hidden_layers_neurons_tuple= (91, 92, 23, 182, 87),max_iter_list = [5,10,20,50,100],activation_list = ['identity','logistic','tanh','relu']):
    for activation in activation_list:
        results = []
        epoch = 0
        for max_iter in max_iter_list:
            model = MLPClassifier(hidden_layer_sizes = hidden_layers_neurons_tuple,max_iter=max_iter,activation=activation,verbose=True)   
            start_t = time.time()
            model.fit(X_train,y_train)
            training_time = time.time()-start_t
            prediction = model.predict(X_test)
            precision = precision_score(y_test,prediction,average ='micro')
            recall = recall_score(y_test,prediction,average = 'micro')
            error = zero_one_loss(y_test,prediction)
            results.append({"max_iter":max_iter,"precision":precision,"error":error,"recall":recall,"training_time":training_time})
        pickle.dump(results, open( "results/by_activation/{}_results.p".format(activation), "wb" )) # Save results
        print("finish for activation {}".format(activation))
    
studyConvergenceDependingActivationFunction(X_train,X_test,y_train,y_test,hidden_layers_neurons_tuple= (91, 92, 23, 182, 87),max_iter_list = [5,10,20,50,100],activation_list = ['identity','logistic','tanh','relu'])

activations = ['identity','logistic','tanh','relu']

for activation in activations:
    results = unpickle(f'results/by_activation/{activation}_results.p')
    precisions = [result["precision"] for result in results]
    max_iter = [result["max_iter"] for result in results]
    plt.plot(max_iter, precisions, label=activation)

plt.xlabel('total_iteration')





plt.ylabel('precision')
plt.title('Evolution of precision by iterations')
plt.legend()
plt.show()


for activation in activations:
    results = unpickle(f'results/by_activation/{activation}_results.p')
    errors = [result["error"] for result in results]
    max_iter = [result["max_iter"] for result in results]
    plt.plot(max_iter, errors, label=activation)

plt.xlabel('total_iteration')
plt.ylabel('error')
plt.title('Evolution of error by iterations')
plt.legend()
plt.show()

for activation in activations:
    results = unpickle(f'results/by_activation/{activation}_results.p')
    recalls = [result["recall"] for result in results]
    max_iter = [result["max_iter"] for result in results]
    plt.plot(max_iter, recalls, label=activation)

plt.xlabel('total_iteration')
plt.ylabel('recall')
plt.title('Evolution of recall by iterations')
plt.legend()
plt.show()

for activation in activations:
    results = unpickle(f'results/by_activation/{activation}_results.p')
    precisions = [result["precision"] for result in results]
    training_time = [result["training_time"] for result in results]
    plt.plot(training_time, precisions, label=activation)

plt.xlabel('training_time(s)')
plt.ylabel('precision')
plt.title('Evolution of precision by training_time')
plt.legend()
plt.show()

# Step 6 

def studyConvergenceDependingAlpha(X_train,X_test,y_train,y_test,hidden_layers_neurons_tuple= (91, 92, 23, 182, 87),max_iter_list = [5,10,20,25], alpha_list = [0.00001,0.0001,0.0005,0.001,0.01]):
    for alpha in alpha_list:
        results = []
        epoch = 0
        for max_iter in max_iter_list:
            model = MLPClassifier(hidden_layer_sizes = hidden_layers_neurons_tuple,max_iter=max_iter,alpha=alpha,verbose=True)   
            start_t = time.time()
            model.fit(X_train,y_train)
            training_time = time.time()-start_t
            prediction = model.predict(X_test)
            precision = precision_score(y_test,prediction,average ='micro')
            recall = recall_score(y_test,prediction,average = 'micro')
            error = zero_one_loss(y_test,prediction)
            results.append({"max_iter":max_iter,"precision":precision,"error":error,"recall":recall,"training_time":training_time})
        pickle.dump(results, open( "results/by_alpha/{}_results.p".format(alpha), "wb" )) # Save results
        print("finish for alpha {}".format(alpha))
    
studyConvergenceDependingAlpha(X_train,X_test,y_train,y_test,hidden_layers_neurons_tuple= (91, 92, 23, 182, 87),max_iter_list = [5,10,20,25],alpha_list = [0.00001,0.0001,0.0005,0.001,0.01])

alphas = [0.00001,0.0001,0.0005,0.001,0.01]

for alpha in alphas:
    results = unpickle(f'results/by_alpha/{alpha}_results.p')
    precisions = [result["precision"] for result in results]
    max_iter = [result["max_iter"] for result in results]
    plt.plot(max_iter, precisions, label=alpha)

plt.xlabel('total_iteration')
plt.ylabel('precision')
plt.title('Evolution of precision by iterations')
plt.legend()
plt.show()


for alpha in alphas:
    results = unpickle(f'results/by_alpha/{alpha}_results.p')
    errors = [result["error"] for result in results]
    max_iter = [result["max_iter"] for result in results]
    plt.plot(max_iter, errors, label=alpha)

plt.xlabel('total_iteration')
plt.ylabel('error')
plt.title('Evolution of error by iterations')
plt.legend()
plt.show()

for alpha in alphas:
    results = unpickle(f'results/by_alpha/{alpha}_results.p')
    recalls = [result["recall"] for result in results]
    max_iter = [result["max_iter"] for result in results]
    plt.plot(max_iter, recalls, label=alpha)

plt.xlabel('total_iteration')
plt.ylabel('recall')
plt.title('Evolution of recall by iterations')
plt.legend()
plt.show()

for alpha in alphas:
    results = unpickle(f'results/by_alpha/{alpha}_results.p')
    precisions = [result["precision"] for result in results]
    training_time = [result["training_time"] for result in results]
    plt.plot(training_time, precisions, label=alpha)

plt.xlabel('training_time(s)')
plt.ylabel('precision')
plt.title('Evolution of precision by training_time')
plt.legend()
plt.show()

