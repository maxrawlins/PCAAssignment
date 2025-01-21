import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.set_printoptions(linewidth=200)


# QUESTION 1
pandas_df = pd.read_csv("stockdata.csv")


# QUESTION 2
print("\n\n\nQUESTION 2\n")


# 2a

#gets all unique names from the dataframe and sorts 
set_of_all_unique_names = pandas_df['Name'].unique() 
sorted_set_of_all_unique_names = sorted(set_of_all_unique_names)

# 2b

# outputs number of values in the sorted set of all unique names
print("number of names: ", len(sorted_set_of_all_unique_names))


# 2c

# outputs first and last 5 names in sorted set of all unique names
print("\nfirst 5 names: ",sorted_set_of_all_unique_names[:5])
print("\nlast 5 names: ",sorted_set_of_all_unique_names[-5:])



# QUESTION 3
print("\n\n\nQUESTION 3\n")

# 3a

# initialise the arrays to store the stocks with either a succicient or insuffiecient amount of data
suffiecient_data_stocks = []  
insuffiecient_data_stocks = []    

#loops through all the names of stocks in the set of all unique names
for name in set_of_all_unique_names:
    # get all records for current name and gets min and max dates
    individual_stock_data = pandas_df[pandas_df['Name'] == name]
    min_date = individual_stock_data['date'].min() 
    max_date = individual_stock_data['date'].max()
    
    # if they have only traded between the two dates they will be added to insufficient_data_stocks
    # to be output in 3b and the others will be stored in sufficient_data stocks
    if min_date <= '2019-11-01' and max_date >= '2022-10-31' :
        suffiecient_data_stocks.append(name)
    else:
        insuffiecient_data_stocks.append(name)


# 3b

print("removed names: ",insuffiecient_data_stocks)


# 3c

#outputs number of stocks that have sufficient amount of data
print("\nnames left: ",len(suffiecient_data_stocks))



# QUESTION 4

print("\n\n\nQUESTION 4\n")


# 4a

# filters dataframe to only contain records with names that appear in suffiecient_data_stocks from q3
suffiecient_data_df = pandas_df[pandas_df['Name'].isin(suffiecient_data_stocks)]
# groups by date and stores count for each of them (the number of records for each date)
record_counts_per_date = suffiecient_data_df.groupby('date').size()
# finds common dates by finding where the count of each date equals the count of suffient_data_stocks
common_dates = record_counts_per_date[record_counts_per_date == len(suffiecient_data_stocks)].index.tolist()


# 4b

# goes through common dates and adds dates that are between 2019-11-01 and 2022-10-31
filtered_common_dates = []
for date in common_dates:
    if '2019-11-01' <= date <= '2022-10-31':
        filtered_common_dates.append(date)

# 4c

#outputs the number of records in the filtered common dates
print("Dates left: ", len(filtered_common_dates))


# 4d

# outputs the first and last 5 dates in filtered common dates
print("\nFirst 5 dates: ", filtered_common_dates[:5])
print("\nLast 5 dates: ", filtered_common_dates[-5:])



# QUESTION 5
print("\n\n\nQUESTION 5\n")


# 5a

# filters the dataframe to only include rows where name is in suffiecient_data_stocks and date is in filtered_common_dates
sufficient_data_stocks_on_valid_days = pandas_df[(pandas_df['Name'].isin(suffiecient_data_stocks)) & (pandas_df['date'].isin(filtered_common_dates))]
# creates dataframe where each row is the date and each column is the stock name and stores the closing price for each stock name on that date
closing_price_df = (sufficient_data_stocks_on_valid_days.groupby(['date', 'Name'])['close'].first().unstack())


#5b
print("closing price data frame:\n\n",closing_price_df)




# QUESTION 6
print("\n\n\nQUESTION 6\n")


# 6a

# create a new datafram to store returns using rows and columns from closing_price_df
stock_returns_df = pd.DataFrame(index=closing_price_df.index[1:], columns=closing_price_df.columns)

#loops through each stock
for stock in closing_price_df.columns:
    # list to store returns for current stock
    stock_returns_list = []
    
    # loop starts at second row because no returns for first date
    for i in range(1, len(closing_price_df)):
        # gets old and new price
        current_close_price = closing_price_df.at[closing_price_df.index[i], stock]
        previous_close_price = closing_price_df.at[closing_price_df.index[i - 1], stock]
        # calculates return using formula and adds to returns list
        stock_returns_list.append((current_close_price - previous_close_price) / previous_close_price )
        
    # adds the stocks returns to the datafrae
    stock_returns_df[stock] = stock_returns_list



# 6b
print("stock returns data frame:\n\n", stock_returns_df)



# QUESTION 7
print("\n\n\nQUESTION 7\n")


# 7a

# initialises pca and fits it to the stock_returns_df
pca = PCA()
pca.fit(stock_returns_df)

# 7b

# outputs top 5 components from pca
print("first 5 components from pca:\n\n", pca.components_[:5])



# QUESTION 8
print("\n\n\nQUESTION 8\n")


# 8a

pca_explained_variance_ratios = pca.explained_variance_ratio_

# 8b

# outputs first ratio as percentage
first_pc_variance_ratio_percentage = pca_explained_variance_ratios[0] * 100
print("1st principal component variance ratio percentage: ", first_pc_variance_ratio_percentage, "%")


# 8c

#creates the graphs size
plt.figure(figsize=(10, 5))
#adds the datapoints and line to the graph
plt.plot(range(1, 21), pca_explained_variance_ratios[:20], marker='o', color ='black', label='explained variance ratio')
#adds label to graph, x axis and y axis
plt.title('first 20 explained variance ratios')
plt.xlabel('principal component')
plt.ylabel('explained variance ratio')
plt.grid(True)


# 8d

# calculates first and second derivatives of explained variance ratios to find elbow
first_pca_derivative = np.diff(pca_explained_variance_ratios)
second_pca_derivative = np.diff(first_pca_derivative)

# gets the index of the elbow and the elbow point(the plus 1s are to account for
# the shortening of the array from the np.diff and converting from zero based to
# 1 based for the point)
pca_elbow_point = np.argmax(second_pca_derivative) + 2  

# anottates the elbow on the graph
plt.axvline(x=pca_elbow_point, color= 'red', linestyle='--', label= f'elbow at PC {pca_elbow_point}')
plt.legend()
plt.show()


 

# QUESTION 9


# 9a

cumulative_pca_variance_ratio = np.cumsum(pca_explained_variance_ratios)


# 9b

#creates the graphs size
plt.figure(figsize=(10,5))
#adds the datapoints and line to the graph
plt.plot(range(1, len(cumulative_pca_variance_ratio) + 1), cumulative_pca_variance_ratio, marker='o', color ='black', label='cumulative variance ratio')
#adds label to graph, x axis and y axis
plt.title('cumulative explained variance ratios')
plt.xlabel('principal component')
plt.ylabel('cumulative variance ratio')
plt.grid(True)


# 9c

#gets number of components to explain 95% variance
num_pc_95 = np.argmax(cumulative_pca_variance_ratio >= 0.95) + 1

# adds x and y lines at 95% variance on graph
plt.axhline(y=0.95, color='blue', linestyle='--', label='95% cumulative variance ratio')
plt.axvline(x=num_pc_95, color='red', linestyle='--', label=f'95% at pc {num_pc_95}')
plt.legend()
plt.show()


# QUESTION 10
print("\n\n\nQUESTION 10\n")


# 10a

normalised_stock_returns = (stock_returns_df - stock_returns_df.mean()) / stock_returns_df.std()



# 10b

# initialises pca_normalised and fits it to the normalised_stock_returns_df
pca_normalised = PCA()
pca_normalised.fit(normalised_stock_returns)

# outputs first 5 components from pca normalise
print("first 5 components from pca normalised:\n", pca_normalised.components_[:5])


# 10c

explained_variance_ratios_normalised = pca_normalised.explained_variance_ratio_

# outputs first normalised ratio as percentage
first_pc_variance_normalised = explained_variance_ratios_normalised[0] * 100
print("\n\n1st normalised principal component variance ratio percentage: ",first_pc_variance_normalised,"%")

#creates the graphs size
plt.figure(figsize=(10, 5))
#adds the datapoints and line to the graph
plt.plot(range(1, 21), explained_variance_ratios_normalised[:20], marker='o', color ='black', label='explained variance ratio')
#adds label to graph, x axis and y axis
plt.title('first 20 normalised explained variance ratios')
plt.xlabel('principal component')
plt.ylabel('explained variance ratio')
plt.grid(True)

# calculates first and second derivatives of explained variance ratios to find elbow
first_derivative_normalised = np.diff(explained_variance_ratios_normalised)
second_derivative_normalised = np.diff(first_derivative_normalised)
# gets the index of the elbow and the elbow point(the plus 1s are to account for
# the shortening of the array from the np.diff and converting from zero based to
# 1 based for the point)
elbow_index_normalised = np.argmax(second_derivative_normalised) + 1  
elbow_point_normalised = elbow_index_normalised + 1  

# anottates the elbow on the graph
plt.axvline(x=elbow_point_normalised, color='red', linestyle='--', label=f'elbow at PC {elbow_point_normalised}')
plt.legend()
plt.show()



# 10d

cumulative_pca_variance_ratios_normalised = np.cumsum(explained_variance_ratios_normalised)

#creates the graphs size
plt.figure(figsize=(10, 5))
#adds the datapoints and line to the graph
plt.plot(range(1, len(cumulative_pca_variance_ratios_normalised) + 1),cumulative_pca_variance_ratios_normalised, marker='o', color ='black',label='cumulative variance ratio')
#adds label to graph, x axis and y axis
plt.title('normalised cumulative variance ratios')
plt.xlabel('principal component')
plt.ylabel('cumulative variance ratio')
plt.grid(True)

#gets number of components to explain 95% variance
num_pc_95_normalised = np.argmax(cumulative_pca_variance_ratios_normalised >= 0.95) + 1

# adds x and y lines at 95% variance on graph
plt.axhline(y=0.95, color='blue', linestyle='--', label='95% cumulative variance ratio')
plt.axvline(x=num_pc_95_normalised, color='red', linestyle='--', label=f'95% at pc {num_pc_95_normalised}')
plt.legend()
plt.show()
