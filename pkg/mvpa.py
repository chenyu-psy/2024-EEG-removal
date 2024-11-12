from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import joblib
import numpy as np


class MVPAMultiClassClassifier:
    def __init__(self, base_classifier):
        """
        Initializes the classifier wrapper.

        Parameters:
        base_classifier: An instance of a scikit-learn classifier to use as the base.
        """
        self.classifier = base_classifier
    
    def train(self, X_train, y_train):
        """
        Trains the base classifier.

        Parameters:
        X_train: Features for training.
        y_train: Labels for training.
        """
        self.classifier.fit(X_train, y_train)
    
    def predict(self, X_test, output_type="label"):
        """
        Generates predictions based on the trained classifier.

        Parameters:
        X_test: Features for prediction.
        output_type: Type of prediction output - "label" or "prob" (probability).

        Returns:
        Predicted labels or class probabilities.
        """
        if output_type == "label":
            return self.classifier.predict(X_test)
        elif output_type == "prob":
            return self.classifier.predict_proba(X_test)
        else:
            raise ValueError("output_type must be 'label' or 'prob'")
    
    def calculate_auc(self, X_test, y_test):
        """
        Calculates the Area Under the Curve (AUC) score for each class.

        Parameters:
        X_test: Features for prediction.
        y_test: True labels for the test set.

        Returns:
        A list containing the AUC score for each class.
        """
        # Get the class probabilities for the test set
        probabilities = self.predict(X_test, output_type="prob")

        # Initialize a list to store AUC scores for each class
        auc_scores = []

        # Create a dictionary to map unique labels to indices
        label_mapping = {label: idx for idx, label in enumerate(sorted(set(y_test)))}
        
        # Convert true labels to numerical labels using the mapping
        numeric_labels = [label_mapping[label] for label in y_test]

        # Calculate the AUC for each class using a one-vs-rest approach
        for class_index in range(probabilities.shape[1]):
            class_probabilities = probabilities[:, class_index]
            
            # Generate binary labels for the current class (one-vs-rest)
            binary_labels = [1 if label == class_index else 0 for label in numeric_labels]

            # Calculate AUC for the current class
            try:
                auc = roc_auc_score(binary_labels, class_probabilities)
            except ValueError:
                auc = float('nan')  # Handle cases where AUC cannot be calculated due to lack of positive/negative samples
            
            # Append the AUC score to the list
            auc_scores.append(auc)
        
        # Return the list of AUC scores for each class
        return auc_scores
    
    def calculate_accuracy(self, X_test, y_test):
        """
        Calculates the accuracy score for each class.

        Parameters:
        X_test: Features for prediction.
        y_test: True labels for the test set.

        Returns:
        A list containing the accuracy score for each class.
        """
        # Get the predicted labels for the test set
        predictions = self.predict(X_test, output_type="label")

        # Create a dictionary to map unique labels to indices
        label_mapping = {label: idx for idx, label in enumerate(sorted(set(y_test)))}
        
        # Convert true labels to numerical labels using the mapping
        numeric_labels = [label_mapping[label] for label in y_test]

        # Initialize a list to store accuracy scores for each class
        accuracy_scores = []

        # Calculate the accuracy for each class using a one-vs-rest approach
        for class_index in range(len(label_mapping)):
            # Generate binary labels for the current class (one-vs-rest)
            binary_labels = [1 if label == class_index else 0 for label in numeric_labels]
            binary_predictions = [1 if pred == class_index else 0 for pred in predictions]

            # Calculate accuracy for the current class
            accuracy = accuracy_score(binary_labels, binary_predictions)
            accuracy_scores.append(accuracy)
        
        # Return the list of accuracy scores for each class
        return accuracy_scores
    
    def calculate_confusion_matrix_accuracy(self, X_test, y_test):
        """
        Calculates the confusion matrix and derives the accuracy for each class.

        Parameters:
        X_test: Features for prediction.
        y_test: True labels for the test set.

        Returns:
        A list containing the accuracy for each class based on the confusion matrix.
        """
        # Get the predicted labels for the test set
        predictions = self.predict(X_test, output_type="label")
        
        # Calculate the confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        # Calculate the accuracy for each class
        class_accuracies = np.divide(cm, cm.sum(axis=1, keepdims=True))
        
        return class_accuracies
    
    def cross_validate(self, X, y, cv=5, n_jobs=-1):
        """
        Performs cross-validation on the model, including AUC and accuracy calculation.

        Parameters:
        X: Features for cross-validation.
        y: Labels for cross-validation.
        cv: Number of cross-validation folds.
        n_jobs: Number of CPU cores to use (-1 means using all available cores).

        Returns:
        A dictionary containing accuracy and AUC scores for each fold.
        """
        kf = KFold(n_splits=cv, shuffle=True)

        def process_fold(train_index, test_index):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train the classifier on the training set
            self.train(X_train, y_train)

            # Calculate accuracy on the test set for each class
            accuracy = self.calculate_accuracy(X_test, y_test)

            # Calculate AUC on the test set for each class
            auc = self.calculate_auc(X_test, y_test)

            # Calculate confusion matrix accuracy on the test set for each class
            cma = self.calculate_confusion_matrix_accuracy(X_test, y_test)
            
            return accuracy, auc, cma

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_fold)(train_index, test_index) for train_index, test_index in kf.split(X)
        )

        accuracy_scores, auc_scores, cma_scores = zip(*results)
        
        return {'accuracy': accuracy_scores, 'auc': auc_scores, 'cma': cma_scores}
    
    def save_classifier(self, save_path):
        """
        Saves the trained classifier to the specified path.

        Parameters:
        save_path: Path to save the trained classifier.
        """
        joblib.dump(self.classifier, save_path)

