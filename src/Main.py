import pandas as pd
import numpy as np
from itertools import permutations
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler


def to_CSV(dataframe, nomDataframe, booleanIndex):
    path = f'../outputs/{nomDataframe}.csv'
    dataframe.to_csv(path, index=booleanIndex)

def prepareDataForEntry(match, numerical_columns, training_columns):
    # This function prepares the data for the entry in the model (encoding, normalizing, etc ...)

    # Encoding categorical features
    match = pd.get_dummies(match, columns=["tournament", "city", "country", "neutral", "home_team", "away_team"])

    # We need to make sure that the one-hot encoded dataframe has the same columns as the training set and same order
    match = match.reindex(columns=training_columns, fill_value=0)

    # Normalizing numerical features
    match[numerical_columns] = sc.transform(match[numerical_columns])
    return match

def getFeatureImportance(randomForestModel, training_columns):
    # Creates a .csv with the feature importance of each feature
    feature_importance = pd.DataFrame(randomForestModel.feature_importances_, index=training_columns,
                                      columns=['importance']).sort_values('importance', ascending=False)

    to_CSV(feature_importance, "feature_importance", True)

def getMatchDataFrame(team1, team2, city, country, numericalColumns, training_columns):
    # This function returns the dataframe of the match between team1 and team2
    # We will have to fetch the data in the dataset (avg goals ...)

    data = {}
    fifa_ranks = pd.read_csv("../databases/fifa_ranking-2023-07-20.csv")
    rera_improved = pd.read_csv("../outputs/rera_improved.csv")


    data['tournament'] = "FIFA World Cup"
    data['city'] = city
    data['country'] = country
    data['neutral'] = True
    data['home_team'] = team1
    data['away_team'] = team2
    data['home_rank'] = fifa_ranks.loc[fifa_ranks['country_full'] == team1, 'rank'].values[-1] # For this one, we fetch the most recent rank of the team in fifa_ranking csv
    data['home_total_points'] = fifa_ranks.loc[fifa_ranks['country_full'] == team1, 'total_points'].values[-1]
    data['home_previous_points'] = fifa_ranks.loc[fifa_ranks['country_full'] == team1, 'previous_points'].values[-1]
    data['home_rank_change'] = fifa_ranks.loc[fifa_ranks['country_full'] == team1, 'rank_change'].values[-1]
    data['away_rank'] = fifa_ranks.loc[fifa_ranks['country_full'] == team2, 'rank'].values[-1]
    data['away_total_points'] = fifa_ranks.loc[fifa_ranks['country_full'] == team2, 'total_points'].values[-1]
    data['away_previous_points'] = fifa_ranks.loc[fifa_ranks['country_full'] == team1, 'previous_points'].values[-1]
    data['away_rank_change'] = fifa_ranks.loc[fifa_ranks['country_full'] == team2, 'rank_change'].values[-1]

    # For the average scores, it's different since we fetch the information from a different dataset
    # which is rera_improved, since it contains all the data we added


    data['home_averageScore'] = rera_improved.loc[rera_improved['home_team'] == team1, 'home_averageScore'].values[-1]

    data['away_averageScore'] = rera_improved.loc[rera_improved['away_team'] == team2, 'away_averageScore'].values[-1]

    if team1 == country:
        # If a team is playing in its own country, it must be team_1 when function called
        data['neutral'] = False


    for i in data:
        # We replace every data by a list containing itself to avoid oncoming problem
        data[i] = [data[i]]

    print("data: ", data)
    matchDF = pd.DataFrame(data)

    return prepareDataForEntry(matchDF, numericalColumns, training_columns)

def parse_championship_file(file_path):
    groups = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Supprimer les sauts de ligne
    lines = [line.strip() for line in lines]

    # Séparer les groupes
    group_names = lines[0].split(';')
    num_groups = len(group_names)

    # Initialiser les groupes
    for group_name in group_names:
        groups[group_name] = []

    # Parcourir les lignes restantes pour ajouter les pays aux groupes
    for line in lines[1:]:
        countries = line.split(';')
        for i in range(num_groups):
            groups[group_names[i]].append({countries[i].lstrip(): 0})

    return groups

def simulateGroupPhase(group, championship):
    # Simulates the matches for a group in the championship
    # Will add the points to the teams in the group
    # +3 if won, +1 if draw, +0 if loss

    # First we get all the matches to play: each country has to play against each other once
    countries = []
    for i in championship[group]:
        for j in i:
            countries.append(j)

    matches_to_play = list(permutations(countries, 2))

    print(matches_to_play)


if __name__ == '__main__':

    ### on lit les databases

    ranking = pd.read_csv("../databases/fifa_ranking-2023-07-20.csv")
    goalscorers = pd.read_csv("../databases/goalscorers.csv")
    results = pd.read_csv("../databases/results.csv")
    shootouts = pd.read_csv("../databases/shootouts.csv")

    ## On déclare la date de début ainsi que la date de fin

    start_date = '1996-01-01'
    end_date = "2022-12-18"

    # Ici la date de la database ranking s'appelle rank_date. On la remplace avec date pour homogéniser avec les autres
    ranking.rename(columns={'rank_date': 'date'}, inplace=True)

    # On filtre par date les databases
    rankings_filtered = ranking[(ranking['date'] >= start_date) & (ranking['date'] <= end_date)]
    goalscorers_filtered = goalscorers[(goalscorers['date'] >= start_date) & (goalscorers['date'] <= end_date)]
    results_filtered = results[(results['date'] >= start_date) & (results['date'] <= end_date)]
    shoothouts_filtered = shootouts[(shootouts['date'] >= start_date) & (shootouts['date'] <= end_date)]

    # On prend que les matchs de Coupe du Monde ou de Qualification à la coupe du Monde
    results_filtered = results_filtered[results_filtered['tournament'].isin(['FIFA World Cup', 'FIFA World Cup qualification'])]

    # On met un peu d'ordre dans la data base rankings, on ordonne premièrement par date pui par classement
    rankings_filtered = rankings_filtered.sort_values(by=['date', 'rank'])



    # On convertit les dates en datetime, pratique pour la suite
    results_filtered['date'] = pd.to_datetime(results_filtered['date'])
    rankings_filtered['date'] = pd.to_datetime(rankings_filtered['date'])

    # On crée daily_ranking qui est une version de rankings_filtered avec un classement par jour
    daily_ranking = rankings_filtered.set_index('date')
    daily_ranking = daily_ranking.groupby('country_full').resample('D').first()
    daily_ranking.ffill(inplace=True)

    # On écrit nos dataframe en csv dans le dossier outputs
    to_CSV(rankings_filtered, "rankings_filtered", False)
    to_CSV(goalscorers_filtered, "goalscorers_filtered", False)
    to_CSV(results_filtered, "results_filtered", False)
    to_CSV(shoothouts_filtered, "shoothouts_filtered", False)

    to_CSV(daily_ranking, "daily_ranking", True)# Ne pas le mettre dans github, le fichier est trop gros

    #2.3 merging Data
    #we will build new dataframe named rera

    dropped_daily_ranking= daily_ranking.drop('country_full', axis=1) #I dropped country_full because there are 2 columns named country_full in daily_ranking
    dropped_daily_ranking=dropped_daily_ranking.reset_index()#date and country_full are now normal columns not indexes anymore


    rera = pd.merge(results_filtered, dropped_daily_ranking, left_on=['home_team', 'date'], right_on=['country_full', 'date'], how='left')

    # Renaming columns  for the home team
    rera = rera.rename(columns={
        'rank': 'home_rank',
        'total_points': 'home_total_points',
        'previous_points': 'home_previous_points',
        'rank_change': 'home_rank_change'
    })

    #Merging the datasets based on away_team and date
    rera = pd.merge(rera, dropped_daily_ranking, left_on=['away_team', 'date'], right_on=['country_full', 'date'],
                         how='left')

    # Renaming columns  for the away team
    rera = rera.rename(columns={
        'rank': 'away_rank',
        'total_points': 'away_total_points',
        'previous_points': 'away_previous_points',
        'rank_change': 'away_rank_change'
    })

    # question 4 (did question 4 before 3 to clean rera and make it easier to add new features)
    #we will do the final tweaking and re-clean the DF after adding all the features of question 3

    # Drop columns which contain redundunt information
    rera = rera.drop(['country_full_x', 'country_full_y', 'country_abrv_x',
                      'confederation_x', 'confederation_y', 'country_abrv_y'], axis=1)
    # drop rows which still contain missing data
    rera.dropna(inplace=True)




    #question 3
    #3.1 adding average number of goals made in the last 5 matches
    #I added two columns home_averageScore and away_averageScore

    copy=rera

    copy['Date'] = pd.to_datetime(rera['date'])


    copy.sort_values(by=['home_team', 'date'], inplace=True)

    # window =5 because we calculate the average during the last 5 matches
    copy['home_averageScore'] = copy.groupby('home_team')['home_score'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    copy.sort_values(by=['away_team', 'date'], inplace=True)
    copy['away_averageScore'] = copy.groupby('away_team')['away_score'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    copy.drop('Date', axis=1, inplace=True) # This column is in double

    to_CSV(copy,"rera_improved",False)


    # ---------------------------------------------------------------------------
    # Random Forest Regressor ---------------------------------------------------
    # ---------------------------------------------------------------------------

    # We will build a random forest regressor using scikit learn.

    # First, we need to prepare the data for the model.
    # We will encode the categorical features (= columns) as one-hot numeric features.
    # TODO! We might apply a "weight" to some of the features, to give more importance to them.
    # We will also normalize the data, to make it easier for the model to learn.

    rf_dataset = copy.copy(deep=True)
    rf_dataset = rf_dataset.drop(["date"], axis=1)
    # I decided to drop the date column because it is very troublesome for the prediction model

    dummies = pd.get_dummies(rf_dataset, columns=["tournament", "city", "country", "neutral", "home_team", "away_team"])

    columnsToPredict = ["home_score", "away_score"]  # What the RF will try to predict

    # Now we separate the data into the training set and the test set
    y = rf_dataset[columnsToPredict]

    columnsToDrop = ["tournament", "city", "country", "neutral", "home_team", "away_team", "home_score", "away_score"]

    X_numeric = rf_dataset.drop(columnsToDrop, axis=1).astype('float64')
    # X_numeric only contains the numeric features (before one-hot encoding)
    # Except the features to predict (the goals scored by each team)

    numericalColumns = X_numeric.columns

    X = pd.concat([X_numeric, dummies], axis=1)

    # Now we can split the data into the training set and the test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
    # We can now normalize the data
    sc = StandardScaler().fit(X_train[numericalColumns])
    X_train[numericalColumns] = sc.transform(X_train[numericalColumns])
    X_test[numericalColumns] = sc.transform(X_test[numericalColumns])

    # The data should be ready for the model that we are going to build and train
    # --------------------------------------------------------------------------


    # Creating our list of parameters, and the model
    params = {"max_depth": [20], "min_samples_split": [10],
              "max_leaf_nodes": [175], "min_samples_leaf": [5],
              "n_estimators": [250], "max_features": ["sqrt"]
              }

    randomForestModel = RandomForestRegressor(random_state=1)
    randomForestGridSearch = GridSearchCV(randomForestModel, params, cv=5, verbose=1, n_jobs=-1)

    # Now we train the model
    randomForestGridSearch.fit(X_train, y_train)
    # We apply to our model the best parameters found by the grid search
    randomForestModel = randomForestGridSearch.best_estimator_

    # Getting the meansquare of the model, features importance etc ...
    y_pred = randomForestModel.predict(X_test)
    print("RFG Mean square error on test set:", mean_squared_error(y_test, y_pred, squared=False))
    getFeatureImportance(randomForestModel, X_train.columns)

    # Prediction d'un match tampon
    # On crée un dataframe avec les données du match, en omettant evidemment les colonnes a predire

    # We create the RF entry by specifying the teams, the city and the country of the match.
    # The function getDataFrameForEntry will fetch the data from the dataset and prepare it for the entry in the model
    match = getMatchDataFrame("Gibraltar", "France", "London", "England", numericalColumns, X_train.columns)

    # We can now predict the score of the match
    match_pred = randomForestModel.predict(match)

    print(f"Home team score: {match_pred[0][0]}, Away team score: {match_pred[0][1]}")


    # ---------------------------------------------------------------------------
    # CHAMPIONSHIP PREDICTION ---------------------------------------------------
    # ---------------------------------------------------------------------------

    # We have .csv file containing the teams and groups of the championship
    # So first we parse it: we WON'T treat it as a dataframe but as a simple text file

    championship = parse_championship_file('../databases/championship.csv')
    city_host = 'London'
    country_host = 'England'

    print(championship)

    # We will now simulate the groups phase
    #for group in championship:
        #simulateGroupPhase(group, championship)

    simulateGroupPhase('Group A', championship)

