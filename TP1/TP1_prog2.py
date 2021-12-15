from sklearn.datasets import fetch_openml 
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.model_selection import KFold
import pickle
import time

def selectRandomData(X,y, size):
    indexes = np.random.randint(y.size, size=size)
    return np.array([X[i] for i in indexes]),np.array([y[i] for i in indexes])

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data

def run_knn(mnist,n_data=5000,repartition=0.8,n_neighbors=10):
    
    print(f'K-nn with k = {n_neighbors} on {n_data} images split into training ({repartition}) et testing set :')

    # Data selection
    X,y = selectRandomData(X=mnist.data,y=mnist.target,size=n_data) # Random selection
    Xtrain,  Xtest,  ytrain,  ytest  =  train_test_split(X,y, train_size=repartition) # Splitting train and test set

    # Training
    print(f'n_neighbors selected = {n_neighbors}')
    clf = neighbors.KNeighborsClassifier(n_neighbors) 
    clf.fit(Xtrain, ytrain) 

    # Predict image 4
    image_4 = Xtest[3]
    plt.imshow(image_4.reshape(28, 28) ,cmap=plt.cm.gray_r,interpolation="nearest") 
    plt.show()
    to_predict = image_4.reshape(1, -1)
    print(f'class = {clf.predict(to_predict)[0]}')

    # Testing
    print(f'Score on all testing set = {clf.score(Xtest, ytest)}')

def find_best_k(mnist,n_data=5000,k_min=2,k_max=15,n_splits=10):

    print(f'K-nn with k between {k_min} et {k_max} on {n_data} images with {n_splits} kfold split')

    # Data selection
    X,y = selectRandomData(X=mnist.data,y=mnist.target,size=n_data) # Random selection

    # Splitting with k-fold
    kf = KFold(n_splits=n_splits,shuffle=True)
    kf.get_n_splits(X)

    # Train and test for different n_neighbors
    scores = []
    for n_neighbors in range(k_min,k_max+1):
        score = 0
        clf = neighbors.KNeighborsClassifier(n_neighbors) 
        
        for train_index, test_index in kf.split(X):
            clf.fit(X[train_index], y[train_index])
            score +=  clf.score(X[test_index], y[test_index])
        scores.append(score/n_splits)

    k_list = [i for i in range(k_min,k_max+1)]
    plt.plot(k_list,scores)
    plt.xlabel('k')
    plt.ylabel('score')
    plt.title('Evolution of score by numbers of neighbors')
    plt.show()

    # Find best k and his score
    best_i = -1
    best_score = -1
    for i,score in enumerate(scores):
        if(best_score<score):
            best_score = score
            best_i = i
    print(f'Best is n_neighbors = {best_i+2} => score = {best_score}')

def varying_proportion_in_training_set(mnist,n_data=5000,k_min=3,k_max=6,N_SPLITS_LIST = [200,500,1000,2000,5000,10000,20000]):
    
    print(f'K-nn with k between {k_min} et {k_max} on {n_data} images with numbers of kfold splits in {N_SPLITS_LIST}')

    # Data selection
    X,y = selectRandomData(X=mnist.data,y=mnist.target,size=n_data) # Random selection

    results = []
    print("testing settings ...")
    for n_splits in N_SPLITS_LIST:
        # Splitting with k-fold
        print(f'\tn_split = {n_splits} <=> proportion train test = {1-1/n_splits}')
        kf = KFold(n_splits=n_splits,shuffle=True)
        kf.get_n_splits(X)

        # Train and test for different n_neighbors
        for n_neighbors in range(k_min,k_max+1):
            score = 0
            clf = neighbors.KNeighborsClassifier(n_neighbors) 

            for train_index, test_index in kf.split(X):
                clf.fit(X[train_index], y[train_index])
                score +=  clf.score(X[test_index], y[test_index])
            results.append({"n_splits":n_splits,"k":n_neighbors,"score":score/n_splits})

    pickle.dump(results, open( "results/results_proportion_training.p", "wb" )) # Save results

    best_score = -1
    best_result = None
    for result in results:
        if(result["score"]>best_score):
            best_score = result["score"]
            best_result = result

    print("Meilleur résultat = {} <=> percent of data in train test = {}".format(best_result,1-1/best_result["n_splits"]))

    # Find best results for each n_splits
    best_score_by_nsplits = np.zeros(np.size(N_SPLITS_LIST))
    for result in results:
        index_n_splits = N_SPLITS_LIST.index(result["n_splits"]) 
        if(result["score"]>best_score_by_nsplits[index_n_splits]):
            best_score_by_nsplits[index_n_splits] = result["score"]

    # Plotting
    PROPORTION_IN_TRAINING_LIST = [1-1/elem for elem in N_SPLITS_LIST]
    plt.plot(PROPORTION_IN_TRAINING_LIST,best_score_by_nsplits)
    plt.xlabel('proportion in training set')
    plt.ylabel('score')
    plt.title('Evolution of score by proportion of data in training set')
    plt.show()

def varying_size_of_dataset(mnist,k_min=3,k_max=6,N_SPLITS_LIST = [20,50,100], N_DATA_LIST = [200,500,1000,2000,5000,10000,20000]):

    print(f'K-nn with k between {k_min} et {k_max} on number of images in N_DATA_LIST with numbers of kfold splits in {N_SPLITS_LIST}')

    results = []
    print("testing settings ...")
    for number_data in N_DATA_LIST:
        print(f'\tnumber of images = {number_data}')
        X,y = selectRandomData(X=mnist.data,y=mnist.target,size=number_data) # Random selection
        
        for n_splits in N_SPLITS_LIST:
        
            # Splitting with k-fold
            print(f'\t\tn_split = {n_splits} <=> proportion train test = {1-1/n_splits}')
            kf = KFold(n_splits=n_splits,shuffle=True)
            kf.get_n_splits(X)

            # Train and test for different n_neighbors
            for n_neighbors in range(k_min,k_max+1):
                score = 0
                clf = neighbors.KNeighborsClassifier(n_neighbors) 

                for train_index, test_index in kf.split(X):
                    clf.fit(X[train_index], y[train_index])
                    score +=  clf.score(X[test_index], y[test_index])
                results.append({"n_data":number_data,"n_splits":n_splits,"n_neighbors":n_neighbors,"score":score/n_splits})

    pickle.dump(results, open( "results/results_size_ds.p", "wb" )) # Save results

    best_score = -1
    best_result = None
    for result in results:
        if(best_score<result["score"]):
            best_score = result["score"]
            best_result = result


    print("Meilleur résultat = {} <=> percent of data in train test = {}".format(best_result,1-1/best_result["n_splits"]))

    # Find best results for each n_splits
    best_score_by_ndata = np.zeros(np.size(N_DATA_LIST))
    for result in results:
        index_n_data = N_DATA_LIST.index(result["n_data"]) 
        if(result["score"]>best_score_by_ndata[index_n_data]):
            best_score_by_ndata[index_n_data] = result["score"]

    # Plotting
    plt.plot(N_DATA_LIST,best_score_by_ndata)
    plt.xlabel('number of data')
    plt.ylabel('score')
    plt.title('Evolution of score by number of data')
    plt.show()

def testing_metrics(mnist, N_NEIGHBORS_LIST = [3,5,8,10,15],n_splits=50,NUMBER_DATA = 20000, metrics = ["euclidean","manhattan","chebyshev","minkowski"]):
    # Parameters

    # Data extraction and selection
    X,y = selectRandomData(X=mnist.data,y=mnist.target,size=NUMBER_DATA) # Random selection

    # Splitting with k-fold

    kf = KFold(n_splits=n_splits,shuffle=True)
    kf.get_n_splits(X)

    # Train and test for different metrics
    for metric in metrics:
            print("KNN with metrics",metric)
            results = []
            for n_neighbors in N_NEIGHBORS_LIST:
                score = 0
                clf = neighbors.KNeighborsClassifier(n_neighbors,metric=metric)
                for train_index, test_index in kf.split(X):
                    clf.fit(X[train_index], y[train_index])
                    score +=  clf.score(X[test_index], y[test_index])
                results.append({"k":n_neighbors,"score":score/n_splits})
                print(f'k : {n_neighbors}\t score : {score/n_splits}')
            pickle.dump(results, open(f'results/by_metrics/results_{metric}.p', "wb" ))

    # plotting
    for metric in metrics:
        results = unpickle(f'results/by_metric/results_{metric}.p')
        scores = [result["score"] for result in results]
        plt.plot(N_NEIGHBORS_LIST, scores, label=metric)

    plt.xlabel('k')
    plt.ylabel('score')
    plt.title('Evolution of score by numbers of neighbors')
    plt.legend()
    plt.show()

def testing_n_jobs(mnist,n_data= 20000,n_neighbors = 5, repartition = 0.5,n_jobs =-1):

    # Data extraction and selection
    X,y = selectRandomData(X=mnist.data,y=mnist.target,size=n_data) # Random selection

    Xtrain,  Xtest,  ytrain,  ytest  =  train_test_split(X,y, train_size=repartition) # Splitting train and test set

    n_jobs = -1
    clf = neighbors.KNeighborsClassifier(n_neighbors,n_jobs=n_jobs) 
    clf.fit(Xtrain, ytrain)
    start_time = time.time()
    score = clf.score(Xtest, ytest) 
    print("time with n_jobs = {} is {} seconds".format(n_jobs,time.time()-start_time))


# Data extraction
print("Extraction of MNIST dataset ...")
mnist = fetch_openml('mnist_784', as_frame=False) # Extraction

# First step
run_knn(mnist,n_data = 5000, repartition = 0.8)
print("\n\n-----------------------------\n\n")
time.sleep(2)

# Second step
find_best_k(mnist,n_data=5000,k_min=2,k_max=15,n_splits=10)
print("\n\n-----------------------------\n\n")
time.sleep(2)

# Third step
varying_proportion_in_training_set(mnist,n_data=5000,k_min=3,k_max=6,N_SPLITS_LIST = [2,3,4,5,8,10,15,20,50,100,500])
print("\n\n-----------------------------\n\n")
time.sleep(2)

# Fourth step
varying_size_of_dataset(mnist,k_min=3,k_max=6,N_SPLITS_LIST = [20,50,100], N_DATA_LIST = [200,500,1000,2000,5000,10000,20000])#
print("\n\n-----------------------------\n\n")
time.sleep(2)

# Fifth step
testing_n_jobs(mnist,n_data= 20000,n_neighbors = 5, repartition = 0.5,n_jobs =-1)
testing_n_jobs(mnist,n_data= 20000,n_neighbors = 5, repartition = 0.5,n_jobs =1)