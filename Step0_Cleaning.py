import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import sklearn.model_selection
from sklearn.impute import SimpleImputer
df = pd.read_csv('heart_disease.csv')
df = pd.DataFrame(df)
df
df.isna().sum(axis='index')
print(df.describe())
df.dtypes
msno.matrix(df, color=(0.5, 0.5, 1))
plt.show()
def cap_outiers_for_column(df, columns, lower_percentile=0.05, upper_percentile=0.95):
    for column in columns:
      if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
        Q1 = df[column].quantile(lower_percentile)
        Q3 = df[column].quantile(upper_percentile)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df
categorical_features = [
    'Gender',
    'Exercise Habits',
    'Smoking',
    'Family Heart Disease',
    'Diabetes',
    'High Blood Pressure',
    'Low HDL Cholesterol',
    'High LDL Cholesterol',
    'Stress Level',
    'Sugar Consumption',
    'Alcohol Consumption',
    'Heart Disease Status'
]
for feature in categorical_features:
    possible_values = df[feature].unique()
    print(f"Possible values for {feature}: {possible_values}")
df_categorical = df[categorical_features]
df_categorical
msno.matrix(df_categorical, color=(0.5, 0.75, 1))
plt.show()
categorical_imputer = SimpleImputer(strategy='most_frequent')
categorical_imputer.fit(df_categorical)
df_categorical = pd.DataFrame(
    data = categorical_imputer.transform(df_categorical),
    columns = categorical_imputer.get_feature_names_out(),
    index = df_categorical.index
)
df_categorical
df_categorical.isna().sum(axis='index')
ordinal_map = {'Low': 0, 'Medium': 1, 'High': 2}
df_categorical['Exercise Habits'] = df_categorical['Exercise Habits'].map(ordinal_map)
df_categorical['Stress Level'] = df_categorical['Stress Level'].map(ordinal_map)
df_categorical['Alcohol Consumption'] = df_categorical['Alcohol Consumption'].map(ordinal_map)
df_categorical['Sugar Consumption'] = df_categorical['Sugar Consumption'].map(ordinal_map)
yesno_map = {'No': 0, 'Yes': 1}
df_categorical['Smoking'] = df_categorical['Smoking'].map(yesno_map)
df_categorical['Diabetes'] = df_categorical['Diabetes'].map(yesno_map)
df_categorical['Family Heart Disease'] = df_categorical['Family Heart Disease'].map(yesno_map)
df_categorical['High Blood Pressure'] = df_categorical['High Blood Pressure'].map(yesno_map)
df_categorical['Low HDL Cholesterol'] = df_categorical['Low HDL Cholesterol'].map(yesno_map)
df_categorical['High LDL Cholesterol'] = df_categorical['High LDL Cholesterol'].map(yesno_map)
yesno_map = {'No': 0, 'Yes': 1}
df_categorical['Smoking'] = df_categorical['Smoking'].map(yesno_map)
df_categorical['Diabetes'] = df_categorical['Diabetes'].map(yesno_map)
df_categorical['Family Heart Disease'] = df_categorical['Family Heart Disease'].map(yesno_map)
df_categorical['High Blood Pressure'] = df_categorical['High Blood Pressure'].map(yesno_map)
df_categorical['Low HDL Cholesterol'] = df_categorical['Low HDL Cholesterol'].map(yesno_map)
df_categorical['High LDL Cholesterol'] = df_categorical['High LDL Cholesterol'].map(yesno_map)
df_categorical
numerical_features = [
    'Age',
    'Blood Pressure',
    'Cholesterol Level',
    'BMI',
    'Sleep Hours',
    'Triglyceride Level',
    'Fasting Blood Sugar',
    'CRP Level',
    'Homocysteine Level'
]
df_numerical = df[numerical_features]
df_numerical
msno.matrix(df_numerical, color=(0.75, 0.75, 1))
plt.show()
numerical_imputer = SimpleImputer(strategy='mean')
numerical_imputer.fit(df_numerical)
df_numerical = pd.DataFrame(
    data = numerical_imputer.transform(df_numerical),
    columns = numerical_imputer.get_feature_names_out(),
    index = df_numerical.index
)
df_numerical
df_numerical.isna().sum(axis='index')
standard_scaler = sklearn.preprocessing.StandardScaler()
standard_scaler.fit(df_numerical)
df_numerical = pd.DataFrame(
    data = standard_scaler.transform(df_numerical),
    columns = standard_scaler.get_feature_names_out(),
    index = df_numerical.index
)
df_numerical
label_encoder = sklearn.preprocessing.LabelEncoder()
label_encoder.fit(df['Heart Disease Status'])
df_label = pd.DataFrame(
    data = label_encoder.transform(df['Heart Disease Status']),
    columns = ['Heart Disease Status'],
    index = df.index
)
df_label
df_categorical = df_categorical.drop(columns=['Heart Disease Status'], errors='ignore')
df = df_categorical.join(df_numerical).join(df_label)
df
df.isna().sum(axis='index')
msno.matrix(df, color=(0.85, 0.75, 1))
plt.show()
HDS = df["Heart Disease Status"].value_counts()
HDS.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#CCCCFF','#fff0b3'], wedgeprops={'edgecolor': '#ffffff', 'linewidth': 1})
plt.title('Heart Disease Status')
plt.ylabel('')
plt.show()
df.to_csv('heart_disease_clean.csv', index=False)
