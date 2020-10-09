import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap

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
def plot_prediction_grid (xx, yy, prediction_grid, confidence_grid, predictors):
    """ 
    Plot KNN predictions for every point on the grid. The background is the prediction of every point of the grid
    and the circular points are the real points of the set and their classifications.
    """
    
    background_colormap = ListedColormap (["hotpink","yellowgreen", "lightskyblue","navajowhite","plum"])
    observation_colormap = ListedColormap (["red","green","blue","darkorange","purple"])
    fig = plt.figure(figsize =(10,10))
    # pcolormesh paints the grid using the prediction grid
    plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5)
    plt.scatter(predictors[:,0], predictors[:,1], c = predictors[:,2], cmap = observation_colormap, s = 35, edgecolor="black", linewidth=0.5)
    plt.xlabel('X coordinate'); plt.ylabel('Y coordinate')
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
    return fig

#Esta es otra
def plot_prediction_confidence_grid(xx, yy, prediction_grid, confidence_grid, predictors, h):
    
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
    background_colormap = ListedColormap (["hotpink","yellowgreen", "lightskyblue","navajowhite","plum"])
    observation_colormap = ListedColormap (["red","green","blue","darkorange","purple"])
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)

    h2 = h/2

    for i in range(len(xx[0]) - 1):
        for j in range(len(yy) - 1):
            rect = Rectangle((xx[0][i]-h2, yy[j][0]-h2), h, h,
                             facecolor=background_colormap(norm(prediction_grid[j, i])), alpha=confidence_grid[j, i], edgecolor='black', linewidth=0.1)
            ax.add_patch(rect)
    
    ax.relim()
    ax.autoscale(enable=True, tight=True)
    ax.scatter(predictors[:,0], predictors[:,1], c = predictors[:,2], cmap = observation_colormap, s = 35, zorder=2, edgecolor="black", linewidth=0.5)

    return fig

def knn_prediction_grid(predictors, k=5, h=0.25, plot=False, plot_format = "normal"):
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
            graph = plot_prediction_grid(xx, yy, prediction_grid, confidence_grid, predictors)
        else:
            graph = plot_prediction_confidence_grid(xx, yy, prediction_grid, confidence_grid, predictors, h)
    return (xx, yy, prediction_grid, confidence_grid, graph)

# Código para testear algoritmo
def split(points, training_prop, test_prop):
    
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
    'plot': Whether you want or not the plot (bool, default = True).
    
    Output:
    'coherences': an np.ndarray containing the coherences values for the differents k values asked for.
    """
    
    coherences = np.array(())
    for i in range(1, lastk + 1):
        coherences = np.append(coherences,[test_knn(training, test, i)])
    return coherences

def plot_multiple_k(coherences):
    fig = plt.figure(figsize=(10,10)) 
    plt.plot(np.array(list(range(1, len(coherences) + 1))), coherences)
    plt.title("KNN con K de 1 a "+str(len(coherences)) )
    plt.xlabel("Valor K")
    plt.ylabel("Coherencia")

    return fig

def determine_best_k(coherences):
    best_coherence = np.amax(coherences)

    bests_k = []
    for i in range(len(coherences)):
        if coherences[i] == best_coherence:
            bests_k.append(i+1)
            
    return bests_k, best_coherence