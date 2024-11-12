from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(test_size=0.3, random_state=42):
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test
