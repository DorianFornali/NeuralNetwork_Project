## NNL_Project
# Universitary project in the first year of Master of Computer Sciences at Université Cote D'Azur.

The goal of the project was to predict the outcome of a football match between two teams or a football championship.
This was achieved using machine learning on two datasets:
- https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
- https://www.kaggle.com/datasets/cashncarry/fifaworldranking

The prediction is done with a Regressor Random Forest or a Multi layer perceptron model of neural network.

# Requirements -
Having installed the following libraries:
    - Pandas
    - Scikitlearn
    - Matplotlib
    - Seaborn

# USAGE:
First, execute the preprocessing script located in the ./pre folder
```
python ./pre/pre_process.py
```
Then inside src folder,
```
python ./Main.py <Country1> <Country2> <City_Host> <Country_Host> <Predictor_Model>
```
--> To predict a match between country 1 and country 2
    Predictor_Model = NN or RF

OR
```
python ./Main.py <path_to_championshipfile> <City_Host> <Country_Host> <Predictor_Model>
```
--> To predict a championship. See databases/championship.csv file for an example
    The countries must appear in the file databases/FIFA_country_list.csv and have the same spelling

# AUTHORS: 
Yasmine Moussaoui
Dorian Fornali
Guillermo Wauquier
