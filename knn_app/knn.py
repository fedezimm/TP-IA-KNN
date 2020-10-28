import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

def generate_random_class_points(mean_x, stdev_x, mean_y, stdev_y, n, label):
    """This function receives as input:
            mean_x: The mean of x coordinate.
            stdev_x: The standard deviation of x coordinate.
            mean_y: The mean of y coordinate.
            stdev_y: The standard deviation of y coordinate.
            n: The number of points of the class.
            label: The label of the class.
        
        Returns: A numpy array of shape (n,3) containing the points in this format [x y classLabel]
    """
    x = np.random.normal(mean_x, stdev_x, n)
    y = np.random.normal(mean_y, stdev_y, n)
    
    return np.array([x, y, np.full(x.shape, label)]).transpose()


def join_groups(groups):
    """This function joins the groups generated in one unique np.array.
       The result of this function is the set 'predictors' for future usage.
       The input must be a tuple containing the groups of points generated befor of shape (n,3) where each point has [x y class]. 
    """
    
    return np.concatenate(groups, axis=0)

def calculate_distance(new_point, arr):
    """
    This function calculates the distance between a set of n points (given) and a new point (also given).
    It returns an erray of shape (n,2) with the results of distances between the new point and each point of the set
    and the class' label. This returned array is already sorted by the distance.
    """

    distance = 0.0
    for i in range(2):
        distance += (new_point[i] - arr[:,i:i+1])**2
        
    distances = np.concatenate((np.sqrt(distance),arr[:,2:3]),axis=1)
    sorted_distances = distances[distances[:,0].argsort()] 
    
    return sorted_distances

def define_class(k,distances):
    """
        This funtion takes the k argument and an ordered distances-array and returns the most repetitive class of the
        first k points (k nearest points)
    """
    
    counts = np.bincount(distances[:k,1].astype(int))
    group = np.argmax(counts)
    confidence = counts[group]/k
    return (group,confidence)


def knn_predict(predictors, point, k):
    """
        This function predicts the class of a new_point given the set of numbers for training and the k argument
    """
    
    sorted_distances = calculate_distance(point, predictors)
    return define_class(k, sorted_distances)

#Código para la grilla
def make_prediction_grid(points, limits, h, k):
    """
        This function takes as input the predictors points of the set (this points already have their class associated
        in the array), then also takes the limits of the future grid, an argument h which is the step number for making
        the grid and the k required to predict each point of the grid with knn algorythm.
        
        
        Ir returns the x coordinates of the grid as xx,
        the y coordinates of the grid as yy,
        the class evaluated for each point of the grid as prediction_grid,
        the confidence of each prediction of the grid as confidence_grid.
    """
    
    x_min, x_max, y_min, y_max = limits
    xs = np.arange(x_min, x_max, h)
    ys = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(xs,ys)
    
    prediction_grid = np.zeros(xx.shape, dtype=int)
    confidence_grid = np.zeros(xx.shape, dtype=float)
    
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            p = np.array([x,y])
            prediction_grid[j,i], confidence_grid[j,i] = knn_predict(points, p, k)
    return (xx, yy, prediction_grid, confidence_grid)

#Todos las funciones que tengan que ver con graficar algo hay que adaptarlo. Esta es una
def plot_prediction_grid (xx, yy, prediction_grid, confidence_grid, predictors, testing,labels):
    """ 
    Plot KNN predictions for every point on the grid. The background is the prediction of every point of the grid
    and the circular points are the real points of the set and their classifications.
    """
    
    background_colormap = ListedColormap (["hotpink","yellowgreen","navajowhite","plum","lightskyblue"])
    fig = plt.figure(figsize =(10,10))
    # pcolormesh paints the grid using the prediction grid
    plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5, shading='nearest')
    cant_class = len(labels)
    dictionary = {
        2:["red","blue"],
        3:["red","darkorange","blue"],
        4:["red","green","darkorange","blue"],
        5:["red","green","darkorange","blueviolet","blue"] 
    }
    observation_colormap = dictionary[cant_class]
    for i in range(cant_class):
        training_points = predictors[predictors[:,2]==i+1]
        testing_points = testing[testing[:,2]==i+1]
        plt.scatter(training_points[:,0], training_points[:,1], facecolors = observation_colormap[i], s = 35, edgecolor="black", linewidth=0.5, label="Training "+str(labels[i]))
        plt.scatter(testing_points[:,0], testing_points[:,1], facecolors = observation_colormap[i], s=60, edgecolor="black", linewidth=0.9, marker="X", label="Testing "+str(labels[i]))
    plt.xlabel('X coordinate'); plt.ylabel('Y coordinate')
    plt.xlim(np.min(xx), np.max(xx))
    plt.ylim(np.min(yy), np.max(yy))
    plt.legend()
    return fig

#Esta es otra
def plot_prediction_confidence_grid(xx, yy, prediction_grid, confidence_grid, predictors, h, testing,labels):
    
    """
    This function plots the grid but now considering the confidence of each prediction. The background now doesn't have
    always the same alpha value (or transparency), More transparency means less confidence on the prediction. More opacity
    means more confidence on the prediction.
    Here, the pcolormesh function cannot take differents values of alpha, so we should construct every single rectangle of 
    the figure and give it a transparency (alpha) value.
    """
    
    from matplotlib.patches import Rectangle, Patch
    from matplotlib.colors import ListedColormap

    norm = plt.Normalize(prediction_grid.min(), prediction_grid.max())
    background_colormap = ListedColormap (["hotpink", "yellowgreen", "navajowhite", "plum", "lightskyblue"])
    observation_colormap = ListedColormap (["red", "green", "darkorange", "blueviolet", "blue"])
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)

    h2 = h/2

    for i in range(len(xx[0]) - 1):
        for j in range(len(yy) - 1):
            rect = Rectangle((xx[0][i]-h2, yy[j][0]-h2), h, h,
                             facecolor=background_colormap(norm(prediction_grid[j, i])), alpha=confidence_grid[j, i], edgecolor='black', linewidth=0.1)
            ax.add_patch(rect)
    
    cant_class = len(labels)
    dictionary = {
        2:["red","blue"],
        3:["red","darkorange","blue"],
        4:["red","green","darkorange","blue"],
        5:["red","green","darkorange","blueviolet","blue"] 
    }
    observation_colormap = dictionary[cant_class]
    for i in range(cant_class):
        training_points = predictors[predictors[:,2]==i+1]
        testing_points = testing[testing[:,2]==i+1]
        ax.scatter(training_points[:,0], training_points[:,1], facecolors = observation_colormap[i], s = 35, edgecolor="black", linewidth=0.5, label="Training "+str(labels[i]),zorder=2)
        ax.scatter(testing_points[:,0], testing_points[:,1], facecolors = observation_colormap[i], s=60, edgecolor="black", linewidth=0.9, marker="X", label="Testing "+str(labels[i]),zorder=2)

    ax.relim()
    ax.set_xlabel('X coordinate'); ax.set_ylabel('Y coordinate');ax.legend()
    ax.autoscale(enable=True, tight=True)
    #ax.scatter(predictors[:,0], predictors[:,1], c = predictors[:,2], cmap = observation_colormap, s = 35, zorder=2, edgecolor="black", linewidth=0.5)
    #ax.scatter(testing[:,0], testing[:,1], c = testing[:,2], cmap = observation_colormap, s=60, zorder=3, edgecolor="black", linewidth=0.9, marker="X")

    return fig

def knn_prediction_grid(predictors, testing, labels,k=5, h=0.25, plot=False, plot_format = "normal"):
    """
        This function takes the predictors, a K parameter (default is 5),
        a step parameter (h) for the grid's steps (default is 0.1),
        the Plot parameter which indicates if it should plot the grid or not
        and the plot_format parameter which indicates wheter the plot shows confidence values or not.
        The possibles values of this parameter are 'normal' (default) and 'confidence'. It takes relevance only
        when the parameter 'plot' is True
        
        It returns:
            xx: x coordinates of the grid.
            yy: y coordinates of the grid.
            prediction_grid: classLabel of the grid.
            confidence_grid: confidence of each prediction.
    """
    if type(plot) != bool:
        raise Exception("The value of the 'plot' parameter must be a boolean.")

    if plot_format != 'normal' and plot_format != "confidence":
        raise Exception("The value of the 'plot_format' parameter must be 'normal'/'confidence'.")
        
    xx, yy, prediction_grid, confidence_grid = make_prediction_grid(predictors, (np.min(predictors[:,0]) - 1.5, np.max(predictors[:,0]) + 1.5, np.min(predictors[:,1]) - 1.5, np.max(predictors[:,1]) + 1.5), h, k)
    if plot:
        if plot_format == "normal":
            graph = plot_prediction_grid(xx, yy, prediction_grid, confidence_grid, predictors, testing, labels)
        else:
            graph = plot_prediction_confidence_grid(xx, yy, prediction_grid, confidence_grid, predictors, h, testing, labels)
    return (xx, yy, prediction_grid, confidence_grid, graph)

# Código para testear algoritmo
def split(points, training_prop, test_prop, seed = 41):
    
    """
    This function takes:
        * the total set of points,
        * the training proportion, which indicates the proportion of the total set that we want for training,
        * the test proportion, which indicates the proportion of the total set that we want for testing.
    
    training_prop + test_prop must be 1, otherwise the function returns an exception.
    
    The result of the function are two differents sets extracted from the total set of points:
        * the training set, and
        * the test set
    """
    
    if training_prop + test_prop != 1:
        raise Exception("ValueError: Proportions of training and test sets must sums 1.")
        
    points_copy = points.copy()
    np.random.seed(seed)
    np.random.shuffle(points_copy)
    
    split_point = round(len(points)*training_prop)
    
    training = points_copy[:split_point]
    test = points_copy[split_point:]
    
    return (training, test)

def test_knn(training, test, k):
    """
        This function takes the training set of points , the test set of points and the parameter K.
        It applies the knn algorythm for every point in the test set and it returns the value of coherence counting 
        the number of times that the prediction is good and then dividing this value above the total number of cases (in the test set).
    """
    
    test_with_predictions = np.concatenate((test, np.zeros((len(test), 2))), axis = 1)
    for point in test_with_predictions:
        point[3], point[4] = knn_predict(training, point, k)
    
    positive_predictions = np.where(test_with_predictions[:,2] == test_with_predictions[:,3])[0].size
    
    return positive_predictions / len(test)

#Acá también adaptar para graficar

def test_multiple_knn(training, test, lastk = 10):
    """
    This function takes the training and test set to obtain the model's coherences for multiple k_values
    and plots the (k, coherence) graphic if needed
    
    Inputs:
    'training': the training set (np.ndarray),
    'test': the test set (np.ndarray),
    'lastk': With how many k values you want to test the model? K is taken from 1 to lastk (integer, default = 10),
    
    Output:
    'coherences': an np.ndarray containing the coherences values for the differents k values asked for.
    """
    
    coherences = np.array(())
    for i in range(1, lastk + 1):
        coherences = np.append(coherences,[test_knn(training, test, i)])
    return coherences


def plot_multiple_k(coherences, ylim=None):
    """
    This function takes the coherences array of multiple k values and plots a "K vs Accuracy" graph 
    """
    fig = plt.figure(figsize=(10,10)) 
    plt.plot(np.array(list(range(1, len(coherences) + 1))), coherences)
    plt.title("KNN con K de 1 a "+str(len(coherences)) )
    plt.xlabel("Valor K")
    plt.ylabel("Coherencia")
    if ylim is not None:
        plt.ylim(ylim)

    return coherences, fig


def determine_best_k(coherences):
    """
    This function takes the coherences array of multiple k values and determine the best 
    value of coherence and to which K value/s belongs.
    """
    best_coherence = np.amax(coherences)

    bests_k = []
    for i in range(len(coherences)):
        if coherences[i] == best_coherence:
            bests_k.append(i+1)
            
    return bests_k, best_coherence

def plot_training_test_distributions(training, test):
    """
    This function takes both sets (training and test) and plots their distributions to demonstrate that they
    behaviour is similar
    """

    fig, ((ax11, ax12, ax13), (ax21, ax22, ax23)) = plt.subplots(2, 3, figsize=(10,10))
    ax11.hist(training[:,0])
    ax12.hist(training[:,1])
    ax13.hist(training[:,2])
    ax21.hist(test[:,0])
    ax22.hist(test[:,1])
    ax23.hist(test[:,2])
    ax11.set_title("Training X")
    ax12.set_title("Training Y")
    ax13.set_title("Training Classes Distr.")
    ax21.set_title("Test X")
    ax22.set_title("Test Y")
    ax23.set_title("Test Classes Distr.")
    return fig

def test_cross_validation(points, k, cv=5, seed=10):
    subsets_size = round(len(points) / cv)
    points_copy = points.copy()
    np.random.seed(seed)
    np.random.shuffle(points_copy)
    subsets = []
    corteMin = 0
    corteMax = subsets_size
    for i in range(cv-1):
        subset = points_copy[corteMin:corteMax]
        subsets.append(subset)
        corteMin = corteMax
        corteMax = corteMax + subsets_size
    subset = points_copy[corteMin:]
    subsets.append(subset)
    accuracies = []
    for i in range(cv):
        test = subsets[i]
        if i == 0:
            training = np.concatenate(subsets[i+1:])
        elif i == cv-1:
            training = np.concatenate(subsets[:i])
        else:
            training1 = np.concatenate(subsets[:i])
            training2 = np.concatenate(subsets[i+1:])
            training = np.concatenate((training1, training2))
        accuracy = test_knn(training, test, k=k)
        accuracies.append(accuracy)
    accuracies = np.array(accuracies)
    return (accuracies, np.mean(accuracies))

def plot_test_cross_validation_results(points, k_max=10, cv=5, seed=10):
    coherences = []
    for i in range(1,k_max+1):
        coherences.append(test_cross_validation(points, i, cv, seed)[1])
    coherences = np.array(coherences)
    fig = plt.figure(figsize=(10,10))
    plt.plot(list(range(1,k_max+1)),coherences)
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.title("Validación cruzada para K de 1 a "+str(k_max))
    plt.ylim((np.min(coherences)-0.1,1.0))
    return coherences, fig

def plot_test_normal_validation(points, train_prop, seed, k_max=25):
    training, test = split(points, train_prop, 1-train_prop, seed)
    coherences = test_multiple_knn(training, test, k_max)
    return plot_multiple_k(coherences,(np.min(coherences)-0.1,1.0))

def get_normal_validation_metrics(points, train_prop, seed=10, k_max=25):
    values_nv, graph_nv = plot_test_normal_validation(points, train_prop, seed, k_max)
    mean_nv = np.mean(values_nv)
    stdev_nv = np.std(values_nv)
    min_nv = np.min(values_nv)
    max_nv = np.max(values_nv)
    percentile25th_nv = np.percentile(values_nv, 25)
    median_nv =np.median(values_nv)
    percentile75th_nv = np.percentile(values_nv, 75)
    output = {
        "Mean": mean_nv,
        "Standard Deviation": stdev_nv,
        "Min Value": min_nv,
        "Max Value": max_nv,
        "Percentile 25th": percentile25th_nv,
        "Median": median_nv,
        "Percentile 75th": percentile75th_nv, 
    }
    return values_nv, graph_nv, output

def get_cross_validation_metrics(points, k_max, cv=5, seed=10):
    values_cv, graph_cv = plot_test_cross_validation_results(points, k_max, cv, seed)
    mean_cv = np.mean(values_cv)
    stdev_cv = np.std(values_cv)
    min_cv = np.min(values_cv)
    max_cv = np.max(values_cv)
    percentile25th_cv = np.percentile(values_cv, 25)
    median_cv =np.median(values_cv)
    percentile75th_cv = np.percentile(values_cv, 75)
    output = {
        "Mean": mean_cv,
        "Standard Deviation": stdev_cv,
        "Min Value": min_cv,
        "Max Value": max_cv,
        "Percentile 25th": percentile25th_cv,
        "Median": median_cv,
        "Percentile 75th": percentile75th_cv,
    }
    return values_cv, graph_cv, output

def get_histogram(values):
    fig = plt.figure()
    plt.hist(values)
    return fig

def get_boxplot(values):
    fig = plt.figure()
    plt.boxplot(values)
    return fig

def get_histogram_boxplot(values):

    fig, ((ax11, ax12)) = plt.subplots(1, 2, figsize=(10,5))
    ax11.hist(values)
    ax12.boxplot(values)
    ax11.set_title("Histograma de coherencias")
    ax12.set_title("Diagrama de caja de coherencias")

    return fig

def get_confusion_matrix_binary(training, test, k, threshold=None):
    test_with_predictions = np.concatenate((test, np.zeros((len(test), 2))), axis = 1)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for point in test_with_predictions:
        point[3], point[4] = knn_predict(training, point, k)
        actual_value = point[2]
        opposite_predicted_value = 1 if point[3] == 2 else 2
        if threshold is not None: 
            prediction = point[3] if point[4]>= threshold else opposite_predicted_value
        else:
            prediction = point[3]
        
        if actual_value == 1 and prediction == 1:
            TP += 1
        elif actual_value == 2 and prediction == 2:
            TN += 1
        elif actual_value == 2 and prediction == 1:
            FP += 1
        elif actual_value == 1 and prediction == 2:
            FN += 1

    confusion_matrix = np.array([[TP, FN],[FP, TN]])
    return confusion_matrix

def get_class_confusion_matrix(training, test, classLabel, k, threshold=None):
    test_with_predictions = np.concatenate((test, np.zeros((len(test), 2))), axis = 1)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for point in test_with_predictions:
        point[3], point[4] = knn_predict(training, point, k)
        actual_value= point[2]
        opposite_predicted_value = classLabel if point[3] != classLabel else 0
        if threshold is not None:
            prediction = point[3] if point[4]>= threshold else opposite_predicted_value
        else:
            prediction = point[3]
        
        if actual_value == classLabel and prediction == classLabel:
            TP += 1
        elif actual_value != classLabel and prediction != classLabel:
            TN += 1
        elif actual_value != classLabel and prediction == classLabel:
            FP += 1
        elif actual_value == classLabel and prediction != classLabel:
            FN += 1

    confusion_matrix = np.array([[TP, FN],[FP, TN]])
    return confusion_matrix

def get_cm_metrics(confusion_matrix):
    TP = confusion_matrix[0][0]
    TN = confusion_matrix[1][1]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]
    sensitivity = TP / (TP+FN)
    accuracy = (TP+TN) / (TP+FN+TN+FP) 
    specificity = TN / (TN+FP)
    ba = (sensitivity+specificity)/2
    precision = TP / (TP+FP)
    npv = TN / (TN+FN)
    miss_rate = FN / (FN+TP)
    fall_out = FP / (FP+TN)
    fdr = FP / (FP+TP)
    false_or = FN / (FN+TN)
    f_score = 2*sensitivity*precision/(sensitivity+precision)
    output = {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "negative_predictive_value": npv,
        "miss_rate": miss_rate,
        "fall_out": fall_out,
        "false_discovery_rate": fdr,
        "false_omission_rate": false_or,
        "accuracy": accuracy,
        "balanced_accuracy": ba,
        "f_score": f_score
    }
    return output

def get_classes_confusion_matrixes(training, test, k):
    classes_quantity = np.unique(training[:,2]).size
    confusion_matrixes = []
    for i in range(classes_quantity):
        confusion_matrix = get_class_confusion_matrix(training, test, i+1, k)
        confusion_matrixes.append(confusion_matrix)
    return np.array(confusion_matrixes)

def get_cm_metrics_multiple_classes(confusion_matrixes):
    n = len(confusion_matrixes)
    cms_metrics = []
    for i in range(n):
        cm_metrics = get_cm_metrics(confusion_matrixes[i])
        cms_metrics.append(cm_metrics)
        
    sensitivity = 0
    accuracy = 0
    specificity = 0
    precision = 0
    npv = 0
    miss_rate = 0
    fall_out = 0
    fdr = 0
    false_or = 0
    
    for cm_metrics in cms_metrics:
        sensitivity += cm_metrics["sensitivity"]  
        specificity += cm_metrics["specificity"]
        precision += cm_metrics["precision"]
        npv += cm_metrics["negative_predictive_value"]
        miss_rate += cm_metrics["miss_rate"]
        fall_out += cm_metrics["fall_out"]
        fdr += cm_metrics["false_omission_rate"]
        false_or += cm_metrics["false_omission_rate"]
        accuracy += cm_metrics["accuracy"]
        
    sensitivity = sensitivity/n
    accuracy = accuracy/n
    specificity = specificity/n
    precision = precision/n
    npv = npv/n
    miss_rate = miss_rate/n
    fall_out = fall_out/n
    fdr = fdr/n
    false_or = false_or/n
    f_score = 2*sensitivity*precision/(sensitivity+precision)
    ba = (sensitivity+specificity)/2
    
    output = {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "negative_predictive_value": npv,
        "miss_rate": miss_rate,
        "fall_out": fall_out,
        "false_discovery_rate": fdr,
        "false_omission_rate": false_or,
        "accuracy": accuracy,
        "balanced_accuracy": ba,
        "f_score": f_score
    }
    return output

def plot_binary_classification_roc_metric(training_set, test_set, k):
    test_with_predictions = np.concatenate((test_set, np.zeros((len(test_set), 2))), axis = 1)
    for point in test_with_predictions:
        point[3], point[4] = knn_predict(training_set, point, k)

    values = np.where(test_with_predictions[:,2]==2, 0,test_with_predictions[:,2])
    probability_scores = [point[4] if point[3] == 1 else (1-point[4]) for point in test_with_predictions]
        
    fpr, tpr, threshold = roc_curve(values, probability_scores)
    roc_auc = auc(fpr, tpr)
    
    fig = plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr,'bo-', label = 'AUC = %0.2f' % roc_auc)
    plt.plot([0,1],[0,1],'r--', label="AUC = 0.5")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("1-Especificidad o Ratio de falsos positivos")
    plt.ylabel("Sensibilidad o Ratio de verdaderos positivos")
    plt.title("Curva ROC de KNN con K = " + str(k))
    plt.legend()
    
    return fig, fpr, tpr, roc_auc, threshold

def plot_multiple_classification_roc_metric(training_set, test_set, k):
    classes_quantity = np.unique(training_set[:,2]).size
    test_with_predictions = np.concatenate((test_set, np.zeros((len(test_set), 2))), axis = 1)
    for point in test_with_predictions:
        point[3], point[4] = knn_predict(training_set, point, k)
    
    fig = plt.figure(figsize=(10,10))
    fprs = []
    tprs = []
    roc_aucs = []
    for i in range(classes_quantity):
        
        values = np.where(test_with_predictions[:,2] != i+1, 0, test_with_predictions[:,2])
        probability_scores = [point[4] if point[3] == i+1 else (1-point[4]) for point in test_with_predictions]

        fpr, tpr, threshold = roc_curve(values, probability_scores, pos_label=i+1)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr,'o-', label = 'AUC = %0.2f (Class ' % roc_auc + str(i+1) + ')')
        
        fprs.append(fpr)
        tprs.append(tpr)
        roc_aucs.append(roc_auc)
        
    plt.plot([0,1],[0,1],'r--', label="AUC = 0.5")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("1-Especificidad o Ratio de falsos positivos")
    plt.ylabel("Sensibilidad o Ratio de verdaderos positivos")
    plt.title("Curva ROC de KNN con K = " + str(k))
    plt.legend()
    return fig, fprs, tprs, roc_aucs

