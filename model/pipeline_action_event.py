# Загружаем библиотеки
import math
from datetime import datetime


import dill.settings
dill.settings['recurse'] = True
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import LocalOutlierFactor

# Создаем функцию для загрузки исходных данных, их обработки и создания конечного датафрейма
# для обработки моделями
def download_data():
    df1 = pd.read_csv('data/ga_hits.csv', low_memory=False)
    df1 = df1.drop_duplicates()
    df2 = pd.read_csv('data/ga_sessions.csv', low_memory=False)
    df2 = df2.drop_duplicates()
    targ = ['sub_car_claim_click',
            'sub_car_claim_submit_click',
            'sub_open_dialog_click',
            'sub_custom_question_submit_click',
            'sub_call_number_click',
            'sub_callback_submit_click',
            'sub_submit_success',
            'sub_car_request_submit_click']
    df1['event_action'] = df1['event_action'].apply(lambda x: 1 if x in targ else 0)
    df3 = pd.merge(left=df1, right=df2, on='session_id', how='inner')
    df3['event_action'] = df3['event_action'].fillna(0).astype(int)
    columns_for_drop = [
        'hit_date',
        'hit_time',
        'hit_number',
        'hit_type',
        'hit_referer',
        'hit_page_path',
        'event_category',
        'event_label',
        'event_value',
        'device_os',
        'utm_keyword',
        'device_model',
        'utm_source',
        'utm_campaign',
        'utm_adcontent',
        'device_screen_resolution',
        'visit_date',
        'session_id',
        'client_id'
    ]
    return df3.drop(columns_for_drop, axis=1)

# Сглаживаем данные
def smoothing_data(df):
    #def calculate_outliers(data):
    q25 = df['visit_number'].quantile(0.25)
    q75 = df['visit_number'].quantile(0.75)
    iqr = q75 - q25
    boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
        #return boundaries_outliers

    #boundaries = calculate_outliers(df['visit_number'])
    df.loc[df['visit_number'] < boundaries[0], 'visit_number'] = math.ceil(boundaries[0])
    df.loc[df['visit_number'] > boundaries[1], 'visit_number'] = math.ceil(boundaries[1])
    return df

def create_new_features(df):
    df['geo_country'] = df['geo_country'].apply(lambda x: 1 if x == "Russia" else 0)
    df['geo_city'] = df['geo_city'].apply(lambda x: 2 if x == 'Moscow' else (1 if x == 'Saint Petersburg' else 0))
    df['utm_medium'] = df['utm_medium'].apply(lambda x: 'none' if x in ['(none)', '(not_set)'] else x)
    df['visit_time'] = pd.to_datetime(df['visit_time'], format='%H:%M:%S')
    df['visit_time'] = df['visit_time'].dt.hour
    return df

def balanced_df(df):
    df_1 = df[df['event_action'] == 1]
    df_0 = df[df['event_action'] == 0].iloc[:50000]
    df_balanced = pd.concat([df_1, df_0], axis=0).sample(frac=1).reset_index(drop=True)
    return df_balanced

def main():
    print('Event Action Prediction Pipeline')

    df = download_data()
    df_balanced = balanced_df(df)
    X = df_balanced.drop('event_action', axis=1)
    y = df_balanced['event_action']

    numerical_selector = make_column_selector(dtype_include=['int64', 'float64'])
    categorical_selector = make_column_selector(dtype_include=['object'])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    transformer = Pipeline(steps=[
        ('remove_outliers', FunctionTransformer(smoothing_data)),
        ('edit_columns', FunctionTransformer(create_new_features))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_selector),
        ('categorical', categorical_transformer, categorical_selector)
    ])

    models = (
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=900),
        MLPClassifier(activation='logistic', hidden_layer_sizes=(256, 128, 64), max_iter=5000)
    )

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('transform', transformer),
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, X, y, cv=4, scoring='roc_auc')
        print(f'model: {type(model).__name__}, roc_auc_mean: {score.mean():.4f}, roc_auc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    best_pipe.fit(X, y)
    with open('event_action_pipe.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'Event action prediction model',
                'author': 'Samuilov Vladimir',
                'version': "without any number because it doesn't make sense",
                'date': datetime.now(),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'roc_auc': best_score
            }
        }, file)

        # Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()