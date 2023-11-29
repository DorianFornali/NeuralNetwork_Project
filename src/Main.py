import pandas as pd
from itertools import combinations
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
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
    # The goal is to create a proper entry for the regressor random forest to predict

    data = {}
    fifa_ranks = pd.read_csv("../outputs/rankings_filtered.csv")
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

    homeTeamAvgScores = rera_improved.loc[rera_improved['home_team'] == team1, 'home_averageScore'].values
    if(len(homeTeamAvgScores) == 0):
        # This means the country has no match data in rera_improved, so we will fill the average goals with 0
        # This is for example the case of Cote d'Ivoire or Sao Tome e Principe
        print(f"{team1} has no match data, filling average goals with 0")
    else:
        data['home_averageScore'] = homeTeamAvgScores[-1]

    awayTeamAvgScores = rera_improved.loc[rera_improved['away_team'] == team2, 'away_averageScore'].values
    if (len(awayTeamAvgScores) == 0):
        print(f"{team2} has no match data, filling average goals with 0")
    else:
        data['away_averageScore'] = awayTeamAvgScores[-1]

    if team1 == country:
        # If a team is playing in its own country, it must be team_1 when function called
        data['neutral'] = False


    for i in data:
        # We replace every data by a list containing itself to avoid oncoming problem
        data[i] = [data[i]]

    matchDF = pd.DataFrame(data)

    return prepareDataForEntry(matchDF, numericalColumns, training_columns)

def parse_championship_file(file_path):
    # This function parses the championship file and returns a dictionary of groups and their points
    groups = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]

    group_names = lines[0].split(';')
    num_groups = len(group_names)

    for group_name in group_names:
        groups[group_name] = []

    for line in lines[1:]:
        countries = line.split(';')
        for i in range(num_groups):
            groups[group_names[i]].append({countries[i].lstrip(): 0})

    return groups

def simulateGroupPhase(group, championship, country_host, numericalColumns, training_columns):
    # Simulates the matches for a group in the championship
    # Will add the points to the teams in the group
    # +3 if won, +1 if draw, +0 if loss


    # First we get all the matches to play: each country has to play against each other once
    countries = []
    for i in championship[group]:
        for j in i:
            countries.append(j)


    matches_to_play = list(combinations(countries, 2)) # Returns all the combinations of 2 countries, without repetition

    # Now we will simulate each match
    for match in matches_to_play:
        team1 = match[0]
        team2 = match[1]

        if(team2 == country_host):
            # If the second team is the host country, we will put it in team1 as requested in the getMatchDataFrame function
            tmp = team1
            team1 = team2
            team2 = team1

        try:
            matchDF = getMatchDataFrame(team1, team2, city_host, country_host, numericalColumns, training_columns)
        except:
            print("ERROR: One of the teams is not in the dataset, watch carefully the spelling")
            print(f"The problem comes from \"{team1}\" or \"{team2}\"")
            exit(1)

        matchResult = randomForestModel.predict(matchDF)
        print(f"GROUP PHASE | {team1} - {team2}")
        print("SCORE: ", matchResult[0][0], " - ", matchResult[0][1])

        # If very slight difference between the scores, we will say it is a draw since we are in the group phase
        deltaScore = matchResult[0][0] - matchResult[0][1]
        if(abs(deltaScore) < 0.02):
            print("Very similar scores, we call it a draw, +1 to both team's score")
            championship = addPoints(team1, championship, 1)
            championship = addPoints(team2, championship, 1)
        else:
            if(deltaScore > 0):
                print(f"{team1} won, +3")
                championship = addPoints(team1, championship, 3)
            else:
                print(f"{team2} won, +3")
                championship = addPoints(team2, championship, 3)

    return championship

def addPoints(country, championship, points):
    # Adds points to a country in the championship
    # Used for the knockout phase
    for group in championship:
        for team in championship[group]:
            if(country in team):
                team[country] += points
                return championship

    return championship

def simulateKnockoutPhase(remainingCountries, city_host, country_host, numericalColumns, trainingColumns, isFirstSetOfMatches):
    # Will simulate all of the matches during the knockout phase recursively
    # by updating the winners and list of matched to play until eventually we reach the finals

    matchesToPlay = []
    # Two possible cases: either we are on the very first set of matches of the knockout phase
    # which mean we apply the A1 vs B2, B1 vs A2 etc ... rule
    # Or we are on the next set of matches, which means we apply W1 vs W2, W3 vs W4 etc ...
    if(isFirstSetOfMatches):
        for i in range(1, len(remainingCountries), 2):
            # A1 plays B2, B1 plays A2 etc ...
            matchesToPlay.append([remainingCountries[i - 1][0], remainingCountries[i][1]])
            matchesToPlay.append([remainingCountries[i - 1][1], remainingCountries[i][0]])
    else:
        for i in range(1, len(remainingCountries), 2):
            # W1 plays W2, W3 plays W4 etc ...
            matchesToPlay.append([remainingCountries[i - 1], remainingCountries[i]])

    remainingCountries = [] # We reset the remainingCountries and will append it by the winners, then call recursively the function on it

    print("MATCHES TO PLAY: ", matchesToPlay)
    print("")
    for i in matchesToPlay:
        winner = ""

        # Prediction of the match
        matchDF = getMatchDataFrame(i[0], i[1], city_host, country_host, numericalColumns, trainingColumns)
        matchResult = randomForestModel.predict(matchDF)

        print(f"KNOCKOUT PHASE | {i[0]} - {i[1]}")
        print("SCORE: ", matchResult[0][0], " - ", matchResult[0][1])

        # We decide what to do according to the results we obtained
        deltaScore = matchResult[0][0] - matchResult[0][1]
        if (abs(deltaScore) < 0.02):
            # If very slight difference between the scores, we will go to the shootouts to determine the winner
            print("Very similar scores, we go to shootouts")
            winner = decideShootoutsWinner(i[0], i[1])
        else:
            if (deltaScore > 0):
                print(f"{i[0]} won, goes to the next stage")
                winner = i[0]
            else:
                print(f"{i[1]} won, goes to the next stage")
                winner = i[1]

        remainingCountries.append(winner)
        print("")
    print("CURRENT PHASE WINNERS: ", remainingCountries)
    print("GOING TO NEXT SET OF MATCHES")

    if(len(remainingCountries) > 1):
        simulateKnockoutPhase(remainingCountries, city_host, country_host, numericalColumns, trainingColumns, False)
    else:
        print("------------------------------------------------------------------")
        print("WE HAVE A WORLD CUP WINNER: ", remainingCountries[0])
        print("------------------------------------------------------------------")


def decideShootoutsWinner(team1, team2):
    # Simulates the shootouts between two teams
    # In fact it is way less complicated than simulating,
    # We will just take a look at the shootouts.csv and take into account the amount of shootouts won by each team
    # and if they already won against each other in the past
    shootoutsCSV = pd.read_csv("../databases/shootouts.csv")

    team1_wins, team2_wins = (0, 0)

    # We fetch in the dataset if such a shootout already happened between the two teams in the past
    # We add the wins according to these previous shootouts
    team1_wins += len(shootoutsCSV.loc[(shootoutsCSV['home_team'] == team1) & (shootoutsCSV['away_team'] == team2) & (shootoutsCSV['winner'] == team1)])
    team1_wins += len(shootoutsCSV.loc[(shootoutsCSV['home_team'] == team2) & (shootoutsCSV['away_team'] == team1) & (shootoutsCSV['winner'] == team1)])

    team2_wins += len(shootoutsCSV.loc[(shootoutsCSV['home_team'] == team1) & (shootoutsCSV['away_team'] == team2) & (shootoutsCSV['winner'] == team2)])
    team2_wins += len(shootoutsCSV.loc[(shootoutsCSV['home_team'] == team2) & (shootoutsCSV['away_team'] == team1) & (shootoutsCSV['winner'] == team2)])

    if(team1_wins == team2_wins):
        # We cannot decide the winner, so we will return the team with the highest rank
        rankings = pd.read_csv("../outputs/rankings_filtered.csv")
        team1_rank = rankings.loc[(rankings['country_full'] == team1), 'rank'].values[-1]
        team2_rank = rankings.loc[(rankings['country_full'] == team2), 'rank'].values[-1]
        if(team1_rank < team2_rank):
            return team1
        else:
            return team2

    if(team1_wins > team2_wins):
        return team1
    else:
        return team2




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
    # match = getMatchDataFrame("Croatia", "France", "London", "England", numericalColumns, X_train.columns)

    # We can now predict the score of the match
    # match_pred = randomForestModel.predict(match)

    # print(f"Home team score: {match_pred[0][0]}, Away team score: {match_pred[0][1]}")


    # ---------------------------------------------------------------------------
    # CHAMPIONSHIP PREDICTION ---------------------------------------------------
    # ---------------------------------------------------------------------------

    # We have .csv file containing the teams and groups of the championship
    # So first we parse it: we WON'T treat it as a dataframe but as a simple text file

    championship = parse_championship_file('../databases/championship.csv')
    city_host = 'London'
    country_host = 'England'

    # WE WILL NOW SIMULATE THE GROUP PHASE --------------------------------------
    for group in championship:
        championship = simulateGroupPhase(group, championship, country_host ,numericalColumns, X_train.columns)

    # Groups phase is over, we select the 2 best teams of each group to go to the knockout phase
    print("GROUP PHRASE RESULTS: ------------------")
    print(championship)

    groupWinners = {}
    for group, teams in championship.items():
        sorted_teams = sorted(teams, key=lambda x: list(x.values())[0], reverse=True)
        groupWinners[group] = [sorted_teams[0], sorted_teams[1]]

    remainingCountries = []
    for i in groupWinners.items():
        # Quite a messy way to extract the country names from the dictionary
        country = i[1][0]
        currentGrpWinners = []
        for j in country.keys():
            currentGrpWinners.append(j)

        country = i[1][1]
        for j in country.keys():
            currentGrpWinners.append(j)

        remainingCountries.append(currentGrpWinners)

        print("WINNERS OF ", i[0], ": ", currentGrpWinners)

    # WE WILL NOW SIMULATE THE KNOCKOUT PHASE --------------------------------------
    # 1st of Group A plays the 2nd of Group B, 1st of Group B plays the 2nd of Group A, etc ...
    # Knockout phase contains all of the sets of matches, from 8th of finals (for example) to the finals

    simulateKnockoutPhase(remainingCountries, city_host, country_host, numericalColumns, X_train.columns, True)
