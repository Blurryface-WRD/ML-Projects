from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_knn_model
from src.model_evaluation import evaluate_model

def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Train the model
    model = train_knn_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
