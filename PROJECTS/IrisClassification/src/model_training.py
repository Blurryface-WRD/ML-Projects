from sklearn.neighbors import KNeighborsClassifier

def train_knn_model(X_train, y_train, n_neighbors=3):
    # Initialize the k-NN classifier
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model
