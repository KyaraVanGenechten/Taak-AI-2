# Importing of libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import category_encoders as ce
from io import StringIO, BytesIO
from PIL import Image
import pydotplus
from sklearn.tree import export_graphviz
from sklearn.svm import SVC  
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Read the CSV file 
health_survey_df = pd.read_csv("resources/age_prediction.csv", sep=',')

# Title and subheaders for the applciation
st.title("Task ML: Benchmarking two ML algorithms")
st.subheader("Information of the dataset")
# Information about the dataset
st.write("I chose a dataset from the national health and nutrition health survey of 2013-2014 about age prediction. I chose this because I find it the most interesting and I would like to learn more about it. I find it interesting to go and see these predictions. \n This is the UCI website link of the dataset: https://archive.ics.uci.edu/dataset/887/national+health+and+nutrition+health+survey+2013-2014+(nhanes)+age+prediction+subset ")
st.subheader("Small EDA on the dataset")

show_dataset_eda = st.button("Show EDA on this dataset")

# Only display the EDA information if the button is clicked
if show_dataset_eda:
    # Display the first rows of the dataset
    st.write("First rows of the dataset:")
    st.write(health_survey_df.head())

    # Display summary statistics
    st.write("Describe the dataset:")
    st.write(health_survey_df.describe())

    # Number of instances and features
    number_instances = health_survey_df.shape[0]
    number_features = health_survey_df.shape[1]
    st.write("Number of Instances:", number_instances)
    st.write("Number of Features:", number_features)

    # Check for missing values
    missing_values = health_survey_df.isnull().sum()
    st.write("Missing values per feature:")
    st.write(missing_values)

    # Checking for unbalanced features
    unique_value_counts = health_survey_df.nunique()
    st.write("Number of unique values per feature:")
    st.write(unique_value_counts)


st.subheader("Different ML techniques from the scikit-learn library")

# A tab for each technique
with st.expander("Decision Tree Classifier"):

    # All features in the variable feature_cols
    feature_cols = ['SEQN', 'age_group', 'RIAGENDR', 'PAQ605', 
                    'BMXBMI', 'LBXGLU', 'DIQ010', 'LBXGLT', 'LBXIN']

    # X and y variables
    X = health_survey_df[feature_cols]
    y = health_survey_df[['RIDAGEYR']]  # target variable

    # Split the dataset into a training set (70%) and a test set (30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    st.write("Dataset training and test information:")
    st.write("X_train shape:", X_train.shape)
    st.write("X_test shape:", X_test.shape)
    st.write("y_train shape:", y_train.shape)
    st.write("y_test shape:", y_test.shape)
    st.write("Result of the Decion Tree Classifier: So as you can see the graph is too large to display. This is because our dataset is very large but also has a lot of columns and is probably a bit of a complex dataset.")

    ce_ord = ce.OrdinalEncoder(cols=feature_cols)
    X_cat = ce_ord.fit_transform(X)

    # Train a Decision Tree Classifier
    clf = DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(X_cat, y)

    class_names = [str(class_name) for class_name in clf.classes_]
    # # Display the graph
    # dot_data = StringIO()
    # export_graphviz(clf, out_file=dot_data, filled=True, rounded=True,
    #                 special_characters=True, feature_names=X.columns, class_names=class_names)
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

    # # Convert the PyDotPlus image to PNG format
    # png_image = Image.open(BytesIO(graph.create_png()))
    # st.image(png_image, width=1000)

    # Accuracy
    y_pred = clf.predict(X_cat)  
    accuracy = accuracy_score(y, y_pred)  
    st.write("Accuracy:", accuracy)
    st.write("This got an accuracy of 1.00. So we can conclude that this is a good technique.")


with st.expander("Support vector machine"):
     # Define your features (X) and target variable (y)
    categorical_features = ['age_group']
    numeric_features = ['SEQN', 'RIDAGEYR', 'RIAGENDR', 'PAQ605', 'BMXBMI', 'LBXGLU', 'DIQ010', 'LBXGLT', 'LBXIN']
    target_col = 'RIDAGEYR'  # Set 'RIDAGEYR' as your target variable for classification

    # Variable X and y
    X = health_survey_df[categorical_features + numeric_features]
    y = health_survey_df[target_col]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a ColumnTransformer to apply preprocessing to categorical and numeric columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_features),
            ('num', StandardScaler(), numeric_features)
        ],
        remainder='passthrough'  # Include other columns as-is
    )

    # Create an SVM classifier (SVC for classification)
    classifier = SVC(kernel='linear', C=1.0)

    # Create a pipeline that combines preprocessing and the SVM classifier
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    # Fit the pipeline to the training data
    pipeline.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = pipeline.predict(X_test)

    # Evaluate the model's performance using accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Accuracy:", accuracy)
    st.write("So we can see that the accuracy of this technique is very low. There may be several reasons why this is very low. An reason can be that the model is very complex,.. ")

with st.expander("Gaussian Naïve Bayes"):
    # Copy the original dataframe to make changes without modifying the original data
    modified_dataframe = health_survey_df.copy()

    # Use LabelEncoder to convert the "age_group" column to numerical values
    le = LabelEncoder()
    modified_dataframe['age_group'] = le.fit_transform(modified_dataframe['age_group'])

    # Define your features (X) and target variable (y)
    feature_cols = ['age_group', 'SEQN', 'RIDAGEYR', 'RIAGENDR', 'PAQ605', 'BMXBMI', 'LBXGLU', 'DIQ010', 'LBXGLT', 'LBXIN']

    X = modified_dataframe[feature_cols]
    y = modified_dataframe['RIDAGEYR']

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Gaussian Naive Bayes classifier
    nb_classifier = GaussianNB()

    # Fit the model to the training data
    nb_classifier.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = nb_classifier.predict(X_test)

    # Calculate the model's accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.write('Accuracy:', accuracy)
    st.write("This technique is very high, it makes perfect predictions for every test data. So the Gaussian Naïve Bayes technique is one of the best")
    # Classification report
    st.image("resources/Report.png", caption="This is the classification report")
    
with st.expander("Comparison"):
    st.write("So with the SVM technique I got an accuracy of 0.2258… and with the Gaussian Naïve Bayes I got an accuracy of 0.99. So we can see that the accuracy of the SVM technique is very low. There may be several reasons why this is very low. The Gaussian Naïve Bayes technique is very high, it makes perfect predictions for every test data. So the Gaussian Naïve Bayes technique and the Decision Tree Classifier are the best and the most accurate of this three. ")


 