import sys
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
import matplotlib.pyplot as plt

import seaborn as sb

def to_CSV(dataframe, nomDataframe, booleanIndex):
    path = f'../outputs/{nomDataframe}.csv'
    dataframe.to_csv(path, index=booleanIndex)

def scoreToPercentage(x1, x2):
    # Used to determine winning probability based on the score of a team in a match
    facteur = 1
    res = 1/ (1 + np.exp(-facteur * (x1 - x2)))
    return (res, 1-res)

def prepareDataForEntry(match, numerical_columns, training_columns):
    # This function prepares the data for the entry in the model (encoding, normalizing, etc ...)

    # Encoding categorical features
    match = pd.get_dummies(match, columns=["tournament", "city", "country", "neutral", "home_team", "away_team"])

    # We need to make sure that the one-hot encoded dataframe has the same columns as the training set and same order
    match = match.reindex(columns=training_columns, fill_value=0)

    # Normalizing numerical features
    ### WE DECIDE NOT TO NORMALIZE THE DATA FOR THE RANDOM FOREST REGRESSOR ###
    #match[numerical_columns] = sc.transform(match[numerical_columns])
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
    data['rank_difference'] = data['home_rank'] - data['away_rank']

    # For the average scores, it's different since we fetch the information from a different dataset
    # which is rera_improved, since it contains all the data we added

    homeTeamAvgScores = rera_improved.loc[rera_improved['home_team'] == team1, 'home_averageScore'].values
    if(len(homeTeamAvgScores) == 0):
        # This means the country has no match data in rera_improved, so we will fill the average goals with 0
        # This is for example the case of Cote d'Ivoire or Sao Tome e Principe
        pass
    else:
        data['home_averageScore'] = homeTeamAvgScores[-1]

    awayTeamAvgScores = rera_improved.loc[rera_improved['away_team'] == team2, 'away_averageScore'].values
    if (len(awayTeamAvgScores) == 0):
        pass
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
        home_team = match[0]
        away_team = match[1]


        try:
            matchDF = getMatchDataFrame(home_team, away_team, city_host, country_host, numericalColumns, training_columns)
            reverseMatchDF = getMatchDataFrame(away_team, home_team, city_host, country_host, numericalColumns, training_columns)
        except:
            print("ERROR: One of the teams is not in the dataset, watch carefully the spelling", file=sys.stderr)
            print(f"The problem comes from \"{home_team}\" or \"{away_team}\"", file=sys.stderr)
            print("For instance, China is referred as \"China PR\" in the dataset", file=sys.stderr)
            print("Please be careful and refer to FIFA_country_list.csv to see the exact spelling of the countries", file=sys.stderr)
            exit(1)

        if(model=="RF"):
            print("----------------------------------Random Forest Regressor Model Executing---------------------------")
            matchResult = randomForestModel.predict(matchDF)
            reverseMatchResult = randomForestModel.predict(reverseMatchDF)
        else:
            print("----------------------------------Neural Network Regressor Model Executing---------------------------")
            matchResult = nn.predict(matchDF)
            reverseMatchResult = nn.predict(reverseMatchDF)



        deltaScoreMatch = matchResult[0][0] - matchResult[0][1]
        deltaScoreReverseMatch = reverseMatchResult[0][0] - reverseMatchResult[0][1]

        if (abs(deltaScoreReverseMatch) > abs(deltaScoreMatch)):
            # Reverse match has a bigger delta
            if (deltaScoreReverseMatch < 0):
                # Home wins
                winner = home_team
            else:
                # Away wins
                winner = away_team

            matchResult[0][0] = reverseMatchResult[0][1]
            matchResult[0][1] = reverseMatchResult[0][0]
            percentages = scoreToPercentage(reverseMatchResult[0][1], reverseMatchResult[0][0])

            if not(percentages[0] > 0.49 and percentages[0] < 0.51):
                championship = addPoints(winner, championship, 3)
        else:
            if (deltaScoreMatch < 0):
                # Away wins
                winner = away_team

            else:
                # Home wins
                winner = home_team
            percentages = scoreToPercentage(matchResult[0][0], matchResult[0][1])
            if not(percentages[0] > 0.49 and percentages[0] < 0.51):
                championship = addPoints(winner, championship, 3)

        print(f"GROUP PHASE | {home_team} - {away_team}")

        print("SCORE: ", matchResult[0][0], " - ", matchResult[0][1])
        print(f"WINNING PROBABILITY: {percentages[0]} Vs {percentages[1]}")

        if (percentages[0] > 0.49 and percentages[0] < 0.51):
            # Draw, we add +1 to both teams
            print("Very similar scores, +1 to both teams")
            championship = addPoints(home_team, championship, 1)
            championship = addPoints(away_team, championship, 1)
        else:
            print(f"{winner} wins the match")

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
        home_team = i[0]
        away_team = i[1]

        matchDF = getMatchDataFrame(home_team, away_team, city_host, country_host, numericalColumns, trainingColumns)
        reverseMatchDF = getMatchDataFrame(away_team, home_team, city_host, country_host, numericalColumns, trainingColumns)

        if(model=="RF"):
            print("----------------------------------Random Forest Regressor Model Executing---------------------------")
            matchResult = randomForestModel.predict(matchDF)
            reverseMatchResult = randomForestModel.predict(reverseMatchDF)
        else:
            print("---------------------------------Neural Network Regressor Model Executing----------------------------")
            matchResult = nn.predict(matchDF)
            reverseMatchResult = nn.predict(reverseMatchDF)



        deltaScoreMatch = matchResult[0][0] - matchResult[0][1]
        deltaScoreReverseMatch = reverseMatchResult[0][0] - reverseMatchResult[0][1]

        if (abs(deltaScoreReverseMatch) > abs(deltaScoreMatch)):
            # Reverse match has a bigger delta
            if (deltaScoreReverseMatch < 0):
                # Home wins
                winner = home_team
            else:
                # Away wins
                winner = away_team
            matchResult[0][0] = reverseMatchResult[0][1]
            matchResult[0][1] = reverseMatchResult[0][0]
            percentages = scoreToPercentage(reverseMatchResult[0][1], reverseMatchResult[0][0])
        else:
            if (deltaScoreMatch < 0):
                # Away wins
                winner = away_team
            else:
                # Home wins
                winner = home_team
            percentages = scoreToPercentage(matchResult[0][0], matchResult[0][1])

        print(f"KNOCKOUT PHASE {home_team} - {away_team}")

        print("SCORE: ", matchResult[0][0], " - ", matchResult[0][1])
        print(f"WINNING PROBABILITY: {percentages[0]} Vs {percentages[1]}")

        if (percentages[0] > 0.49 and percentages[0] < 0.51):
            # Draw, we will simulate the shootouts
            print("Very similar scores, we go to shootouts")
            winner = decideShootoutsWinner(home_team, away_team)

        print(f"{winner} wins the match")
        remainingCountries.append(winner)
        print("")
    print("CURRENT PHASE WINNERS: ", remainingCountries)
    print("GOING TO NEXT SET OF MATCHES")

    if(len(remainingCountries) > 1):
        return simulateKnockoutPhase(remainingCountries, city_host, country_host, numericalColumns, trainingColumns, False)
    else:
        return remainingCountries[0]


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

    try:
        f = open("../outputs/rera_improved.csv")
        f.close()
    except:
        print("ERROR: You need to run the pre_process.py script first. It is located in pre folder", file=sys.stderr)
        exit(1)

    # Basic usage guide for command-line
    program_arguments = sys.argv


    if(len(program_arguments) not in [5, 6]):
        print("ERROR: Wrong number of arguments", file=sys.stderr)
        print("Usage: python3 Main.py <path_to_championship_csv> <city_host> <country_host> <model_to_use (RF / NN)>",file=sys.stderr)
        print("Example: python3 Main.py ../databases/championship.csv London England RF", file=sys.stderr)
        print("Usage: python3 Main.py <home_team> <away_team> <city_host> <country_host> <model_to_use (RF / NN)>", file=sys.stderr)
        print("Example: python3 Main.py France Germany London England NN", file=sys.stderr)

        exit(1)

    if(len(program_arguments) == 5):
        # We predict a championship, we verify that the given file is correct before data processing,
        # not to loose useless time if the file is wrong

        championship_path = program_arguments[1]
        try:
            championship = parse_championship_file(championship_path)
        except:
            print("ERROR: Wrong path to championship file", file=sys.stderr)

            print("Usage: python3 Main.py <path_to_championship_csv> <city_host> <country_host>", file=sys.stderr)
            print("Example: python3 Main.py ../databases/championship.csv London England", file=sys.stderr)
            print("Usage: python3 Main.py <home_team> <away_team> <city_host> <country_host>", file=sys.stderr)
            print("Example: python3 Main.py France Germany London England", file=sys.stderr)
            exit(1)


    # ---------------------------------------------------------------------------
    # Random Forest Regressor ---------------------------------------------------
    # ---------------------------------------------------------------------------

    # We will build a random forest regressor using scikit learn.

    # First, we need to prepare the data for the model.
    # We will encode the categorical features (= columns) as one-hot numeric features.
    # We will also normalize the data, to make it easier for the model to learn.

    rf_dataset = pd.read_csv("../outputs/rera_improved.csv")
    rf_dataset = rf_dataset.drop(["date"], axis=1)
    #rf_dataset = rf_dataset.drop(["home_averageScore"], axis=1)
    #rf_dataset = rf_dataset.drop(["away_averageScore"], axis=1)

    # I decided to drop the date column because it is very troublesome for the prediction model

    dummies = pd.get_dummies(rf_dataset, columns=["tournament", "city", "country", "neutral", "home_team", "away_team"])

    columnsToPredict = ["home_score", "away_score"]  # What the RF will try to predict

    # Now we separate the data into the training set and the test set
    y = rf_dataset[columnsToPredict]

    columnsToDrop = ["tournament", "city", "country", "neutral", "home_team", "away_team", "home_score",
                     "away_score", "rank_difference"]

    X_numeric = rf_dataset.drop(columnsToDrop, axis=1).astype('float64')
    # X_numeric only contains the numeric features (before one-hot encoding)
    # Except the features to predict (the goals scored by each team)

    numericalColumns = X_numeric.columns

    X = pd.concat([X_numeric, dummies], axis=1)

    # Now we can split the data into the training set and the test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    # We can now normalize the data
    ### WE DECIDE NOT TO NORMALIZE THE DATA FOR THE RANDOM FOREST REGRESSOR ###
    #sc = StandardScaler().fit(X_train[numericalColumns])
    #X_train[numericalColumns] = sc.transform(X_train[numericalColumns])
    #X_test[numericalColumns] = sc.transform(X_test[numericalColumns])

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
    RF_MSE = mean_squared_error(y_test, y_pred)

    getFeatureImportance(randomForestModel, X_train.columns)

    # ---------------------------------------------------------------------------
    # Neural Network Regressor ---------------------------------------------------
    # ---------------------------------------------------------------------------
    nn_dataset = rf_dataset

    # let's check The correlation between the features
    numeric_columns = nn_dataset.select_dtypes(include=[np.number])
    C_mat = numeric_columns.corr()

    fig = plt.figure(figsize=(10, 10))

    sb.heatmap(C_mat, vmax=.8, square=True)
    #plt.show()#decomment this line if you want to see correlation


    #---------------------------grid_searchCV phase
    #this part of the code takes a long time to excute so I decided to manually print the best params just once and then
    #using them directly in my function
    # Create an MLPRegressor instance
    mlp_regressor = MLPRegressor()

    # Define the parameter grid
    param_grid = {
        'hidden_layer_sizes': [(100, 100,),(50,50)],
        'activation': ['logistic'],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter': [900, 1000, 1100]

    }

    # Create GridSearchCV
    #grid_search = GridSearchCV(mlp_regressor, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    # Fit the model to the data
    #grid_search.fit(X_train, y_train)

    # Print the best parameters and corresponding score
    # print("Best Parameters: ", grid_search.best_params_)
    # defining our model
    nn = MLPRegressor(hidden_layer_sizes=(100, 100,), activation='logistic', max_iter=1000)
    # Now we train the model
    nn.fit(X_train, y_train)
    # calculating the errors
    mae = metrics.mean_absolute_error(y_train, nn.predict(X_train))
    mse = metrics.mean_squared_error(y_train, nn.predict(X_train))
    rsq = metrics.r2_score(y_train, nn.predict(X_train))







    if(len(program_arguments) == 5):
        # We used the script to predict a championship
        city_host = program_arguments[2]
        country_host = program_arguments[3]
        model = program_arguments[4]

        # ---------------------------------------------------------------------------
        # CHAMPIONSHIP PREDICTION ---------------------------------------------------
        # ---------------------------------------------------------------------------

        # We have .csv file containing the teams and groups of the championship
        # So first we parse it: we WON'T treat it as a dataframe but as a simple text file


        # WE WILL NOW SIMULATE THE GROUP PHASE --------------------------------------
        # ---------------------------------------------------------------------------

        for group in championship:
            championship = simulateGroupPhase(group, championship, country_host ,numericalColumns, X_train.columns)

        # Groups phase is over, we select the 2 best teams of each group to go to the knockout phase
        print("GROUP PHASE OVER ------------------")
        print("------------------------------------")

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
        # Knockout phase contains all the sets of matches, from 8th of finals (for example) to the finals

        champion = simulateKnockoutPhase(remainingCountries, city_host, country_host, numericalColumns, X_train.columns, True)

        print("------------------------------------------------------------------")
        print("WE HAVE A WORLD CUP WINNER: ", champion)
        print("------------------------------------------------------------------")

        print("\nGROUP PHASE RESULTS:")
        for i in championship.items():
            print(i[0], ": ", i[1])

    if(len(program_arguments) == 6):
        # ---------------------------------------------------------------------------
        # PREDICTION OF A SINGLE MATCH ----------------------------------------------
        # ---------------------------------------------------------------------------


        # In that use-case of the script, we want to predict
        # the result of a match between two teams
        home_team = program_arguments[1]
        away_team = program_arguments[2]
        city_host = program_arguments[3]
        country_host = program_arguments[4]
        model = program_arguments[5]

        # We predict the match but in both side, A - B and B - A
        # Because the model seems to be biased towards the home team
        # And we take the biggest difference in score to deduce the winner
        matchDF = getMatchDataFrame(home_team, away_team, city_host, country_host, numericalColumns, X_train.columns)
        reverseMatchDF = getMatchDataFrame(away_team, home_team, city_host, country_host, numericalColumns, X_train.columns)

        if(model=="RF"):
            print("--------------Random Forest Model executing------------------")
            matchResult = randomForestModel.predict(matchDF)
            reverseMatchResult = randomForestModel.predict(reverseMatchDF)
        else:
            print("--------------Neural Network Model executing------------------")
            matchResult = nn.predict(matchDF)
            reverseMatchResult = nn.predict(reverseMatchDF)




        deltaScoreMatch = matchResult[0][0] - matchResult[0][1]
        deltaScoreReverseMatch = reverseMatchResult[0][0] - reverseMatchResult[0][1]

        if(abs(deltaScoreReverseMatch) > abs(deltaScoreMatch)):
            # Reverse match has a bigger delta
            if(deltaScoreReverseMatch < 0):
                # Home wins
                winner = home_team
            else:
                # Away wins
                winner = away_team
            matchResult[0][0] = reverseMatchResult[0][1]
            matchResult[0][1] = reverseMatchResult[0][0]
            percentages = scoreToPercentage(reverseMatchResult[0][1], reverseMatchResult[0][0])
        else:
            if(deltaScoreMatch < 0):
                # Away wins
                winner = away_team
            else:
                # Home wins
                winner = home_team
            percentages = scoreToPercentage(matchResult[0][0], matchResult[0][1])

        print(f"{home_team} - {away_team}")

        print("SCORE: ", matchResult[0][0], " - ", matchResult[0][1])
        print(f"WINNING PROBABILITY: {percentages[0]} Vs {percentages[1]}")

        if(percentages[0] > 0.49 and percentages[0] < 0.51):
            # Draw, we will simulate the shootouts
            print("Very similar scores, we go to shootouts")
            winner = decideShootoutsWinner(home_team, away_team)

        print(f"{winner} wins the match")

    print("RF Model's score on training data:", randomForestModel.score(X_train, y_train))
    print("RF Model's score on testing data:", randomForestModel.score(X_test, y_test))
    print("RF Model's Mean Square Error on training data:", RF_MSE)

    print("Neural Network's score on training data:", nn.score(X_train, y_train))
    print("Neural Network's score on testing data:", nn.score(X_test, y_test))
    print("Neural Network's Mean Square Error on training data:", mse)

    exit(0)