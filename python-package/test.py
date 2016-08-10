def dummydata():
    '''
        Generate a few data points, with 2 features
    '''
    N_POINTS = 10 # Points for each class
    N_FEATURES = 3
    N_CLASSES = 2

    X = np.zeros((N_CLASSES * N_POINTS, N_FEATURES))
    y = np.zeros((N_CLASSES * N_POINTS))

    for i in range(N_CLASSES):
        X[i * N_POINTS: (i + 1) * N_POINTS] = np.random.normal([0,i,0], [1,1,1], (N_POINTS, N_FEATURES))
        y[i * N_POINTS: (i + 1) * N_POINTS] = i * np.ones((N_POINTS))

    return X,y

