import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


MONTHS = {
    'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3,
    'May': 4, 'June': 5, 'Jul': 6, 'Aug': 7,
    'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11
}


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into
    a list of evidence lists and a list of labels. Return (evidence, labels).
    """
    evidence = []
    labels = []
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # build evidence row
            e = []
            # integer fields
            e.append(int(row['Administrative']))
            e.append(float(row['Administrative_Duration']))
            e.append(int(row['Informational']))
            e.append(float(row['Informational_Duration']))
            e.append(int(row['ProductRelated']))
            e.append(float(row['ProductRelated_Duration']))
            e.append(float(row['BounceRates']))
            e.append(float(row['ExitRates']))
            e.append(float(row['PageValues']))
            e.append(float(row['SpecialDay']))
            # month
            e.append(MONTHS[row['Month']])
            # integer fields
            e.append(int(row['OperatingSystems']))
            e.append(int(row['Browser']))
            e.append(int(row['Region']))
            e.append(int(row['TrafficType']))
            # visitor type
            e.append(1 if row['VisitorType'] == 'Returning_Visitor' else 0)
            # weekend
            e.append(1 if row['Weekend'] == 'TRUE' else 0)
            evidence.append(e)
            # label
            labels.append(1 if row['Revenue'] == 'TRUE' else 0)
    return evidence, labels


def train_model(evidence, labels):
    """
    Given evidence and labels, return a fitted k-NN (k=1) classifier.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given actual labels and predicted labels, return (sensitivity, specificity).
    """
    true_positive = 0
    total_positive = 0
    true_negative = 0
    total_negative = 0
    for actual, predicted in zip(labels, predictions):
        if actual == 1:
            total_positive += 1
            if predicted == 1:
                true_positive += 1
        else:
            total_negative += 1
            if predicted == 0:
                true_negative += 1
    sensitivity = true_positive / total_positive if total_positive else 0
    specificity = true_negative / total_negative if total_negative else 0
    return sensitivity, specificity


if __name__ == "__main__":
    main()
