
# Importations
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
    
#-----------------------------------------------------------------------
def downcast_dtypes(dataframe, save=False):
    '''This function tries to downcast integer and floating dtypes columns
    to the smallest numerical corresponding dtype.
    It returns a dictionnary of the actually downcasted dtypes.'''
    
    import pandas as pd
    
    # initialise the dict of downcasted dtypes for features
    dict_dtypes = {}
    
    # getting list of integer columns
    columns_int = dataframe.select_dtypes(include=['integer']).columns
    
    for column in columns_int:
        old_dtype = str(dataframe[column].dtypes)
        # trying to downcast integer columns (np.int8 to np.int64)
        dataframe[column] = pd.to_numeric(dataframe[column], downcast='integer')
        new_dtype = str(dataframe[column].dtypes)
        
        # if dtype was downcasted
        if new_dtype != old_dtype:
            print("Column {} downcasted from {} to {}.".format(column, old_dtype, new_dtype))
            # add new key in dictionnary
            dict_dtypes[column] = str(dataframe[column].dtypes)

    # getting list of floatting columns
    columns_float = dataframe.select_dtypes(include=['floating']).columns
    
    for column in columns_float:
        old_dtype = str(dataframe[column].dtypes)
        # trying to downcast float columns (np.float32 to np.float64)
        dataframe[column] = pd.to_numeric(dataframe[column], downcast='float')
        new_dtype = str(dataframe[column].dtypes)
        
        # if dtype was downcasted
        if new_dtype != old_dtype:
            print("Column {} downcasted from {} to {}.".format(column, old_dtype, new_dtype))
            # add new key in dictionnary
            dict_dtypes[column] = str(dataframe[column].dtypes)
        
    # Saving as a json file
    if save==True:
        import json
        
        filename = 'dict_dtypes.json'
        with open(filename, 'w') as outfile:
            json.dump(dict_dtypes, outfile)
        
    # return dict of downcasted dtypes
    return dict_dtypes

#-----------------------------------------------------------------------
def global_filling_rate(dataframe):
    """Compute and displays global filling rate of a DataFrame"""

    # get the numbers of rows and columns in the dataframe
    nb_rows, nb_columns = dataframe.shape
    print("DataFrame has {} rows and {} columns.".format(nb_rows, nb_columns))

    # get the number of non-Nan data in the dataframe
    nb_data = dataframe.count().sum()

    # computing the filling rate
    filling_rate = nb_data / (nb_rows * nb_columns)
    missing_rate = 1 - filling_rate

    # computing the total missing values
    missing_values = (nb_rows * nb_columns) - nb_data

    # display global results
    print("")
    print("Global filling rate of the DataFrame: {:.2%}".format(filling_rate))
    print("Missing values in the DataFrame: {} ({:.2%})"
          .format(missing_values, missing_rate))

    # compute number of rows with missing values
    mask = dataframe.isnull().any(axis=1)
    rows_w_missing_values = len(dataframe[mask])
    rows_w_missing_values_percentage = rows_w_missing_values / nb_rows

    # display results
    print("")
    print("Number of rows with missing values: {} ({:.2%})"
          .format(rows_w_missing_values, rows_w_missing_values_percentage))

    # compute number of columns with missing values
    mask = dataframe.isnull().any(axis=0)
    cols_w_missing_values = len(dataframe[dataframe.columns[mask]].columns)
    cols_w_missing_values_percentage = cols_w_missing_values / nb_columns

    # display results
    print("Number of columns with missing values: {} ({:.2%})"
          .format(cols_w_missing_values, cols_w_missing_values_percentage))

    
#-----------------------------------------------------------------------
def columns_filling_rate(dataframe, columns='all', missing_only=False):
    """Calculate and displays the filling rate for
    a particular column in a pd.DataFrame."""
    
    # Importations
    import pandas as pd
    import numpy as np
    
    # If 'feature' is not specified
    if columns == 'all':
        columns = dataframe.columns
        
    
    # initialization of the results DataFrame
    results = pd.DataFrame(columns=['nb_values', 'missing_values', 'filling_rate'])
        
    # for each feature
    for column in columns:

        # Count of the values on each column
        values_count = dataframe[column].count()
        
        # Computing missing values
        nb_rows = dataframe.shape[0]
        missing_values = nb_rows - values_count

        # Computing filling rates
        filling_rate = values_count / nb_rows
        if missing_only and missing_values == 0:
            filling_rate = np.nan
        
        # Adding a row in the results' dataframe
        results.loc[column] = [values_count, missing_values, filling_rate]

    # Sorting the features by number of missing_values
    results = results.dropna(subset=['filling_rate'])
    results = results.sort_values('filling_rate')
    
    if results.empty == False:
        return results
    else:
        print("No missing value.")

#-----------------------------------------------------------------------
def categorical_matrix(dataframe):
    """Displays a kind of "correlation matrix" including categorical features."""

    # loading library
    from dython.nominal import associations, cluster_correlations

    # Get the categorical and boolean columns
    categorical_columns = list(dataframe.select_dtypes(include='category').columns)
    categorical_columns += list(dataframe.select_dtypes(include='bool').columns)

    # Drop NaN values to avoid errors
    df_for_correlations = dataframe.dropna()
    
    # Drop 'object', 'datetime' and 'timedelta' columns
    df_for_correlations = df_for_correlations.select_dtypes(exclude=['object', 'datetime', 'timedelta'])

    # Calculate associations and display graph
    assoc = associations(df_for_correlations.select_dtypes(exclude=['object','timedelta64[ns]', 'datetime64[ns]' ]),
             nan_strategy ='drop_samples',
             figsize=(30,30),
             plot=False
             )
    
    # Sort the correlations_matrix s and display graph
    correlations_matrix = assoc['corr']
    correlations_matrix, _ = cluster_correlations(correlations_matrix)
    return (correlations_matrix)

#-----------------------------------------------------------------------

def heatmap(x, y, **kwargs):
    """heatmap with a better readability."""

    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = [1]*len(x)

    if 'palette' in kwargs:
        palette = kwargs['palette']
        n_colors = len(palette)
    else:
        n_colors = 256 # Use 256 colors for the diverging color palette
        palette = sns.color_palette("Blues", n_colors) 

    if 'color_range' in kwargs:
        color_min, color_max = kwargs['color_range']
    else:
        color_min, color_max = min(color), max(color) # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    def value_to_color(val):
        if color_min == color_max:
            return palette[-1]
        else:
            val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            ind = int(val_position * (n_colors - 1)) # target index in the color palette
            return palette[ind]

    if 'size' in kwargs:
        size = kwargs['size']
    else:
        size = [1]*len(x)

    if 'size_range' in kwargs:
        size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]
    else:
        size_min, size_max = min(size), max(size)

    size_scale = kwargs.get('size_scale', 500)

    def value_to_size(val):
        if size_min == size_max:
            return 1 * size_scale
        else:
            val_position = (val - size_min) * 0.99 / (size_max - size_min) + 0.01 # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            return val_position * size_scale
    if 'x_order' in kwargs: 
        x_names = [t for t in kwargs['x_order']]
    else:
        x_names = [t for t in sorted(set([v for v in x]))]
    x_to_num = {p[1]:p[0] for p in enumerate(x_names)}

    if 'y_order' in kwargs: 
        y_names = [t for t in kwargs['y_order']]
    else:
        y_names = [t for t in sorted(set([v for v in y]))]
    y_to_num = {p[1]:p[0] for p in enumerate(y_names)}

    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1) # Setup a 1x10 grid
    ax = plt.subplot(plot_grid[:,:-1]) # Use the left 14/15ths of the grid for the main plot

    marker = kwargs.get('marker', 's')

    kwargs_pass_on = {k:v for k,v in kwargs.items() if k not in [
         'color', 'palette', 'color_range', 'size', 'size_range', 'size_scale', 'marker', 'x_order', 'y_order'
    ]}

    ax.scatter(
        x=[x_to_num[v] for v in x],
        y=[y_to_num[v] for v in y],
        marker=marker,
        s=[value_to_size(v) for v in size], 
        c=[value_to_color(v) for v in color],
        **kwargs_pass_on
    )
    ax.set_xticks([v for k,v in x_to_num.items()])
    ax.set_xticklabels([k for k in x_to_num], rotation=45, horizontalalignment='right')
    ax.set_yticks([v for k,v in y_to_num.items()])
    ax.set_yticklabels([k for k in y_to_num])

    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_facecolor('#F1F1F1')

    # Add color legend on the right side of the plot
    if color_min < color_max:
        ax = plt.subplot(plot_grid[:,-1]) # Use the rightmost column of the plot

        col_x = [0]*len(palette) # Fixed x coordinate for the bars
        bar_y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars

        bar_height = bar_y[1] - bar_y[0]
        ax.barh(
            y=bar_y,
            width=[5]*len(palette), # Make bars 5 units wide
            left=col_x, # Make bars start at 0
            height=bar_height,
            color=palette,
            linewidth=0
        )
        ax.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
        ax.grid(False) # Hide grid
        ax.set_facecolor('white') # Make background white
        ax.set_xticks([]) # Remove horizontal ticks
        ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3)) # Show vertical ticks for min, middle and max
        ax.yaxis.tick_right() # Show vertical ticks on the right 

#-----------------------------------------------------------------------
def corrplot(data, size_scale=500, marker='s'):
    """get corr of a dataframe and output my improve heatmap."""
    corr = pd.melt(data.reset_index(), id_vars='index')
    corr.columns = ['x', 'y', 'value']
    heatmap(
        corr['x'], corr['y'],
        color=corr['value'], color_range=[-1, 1],
        palette=sns.diverging_palette(20, 220, n=256),
        size=corr['value'].abs(), size_range=[0,1],
        #size=size, size_range=[0,1],
        marker=marker,
        x_order=data.columns,
        y_order=data.columns[::-1],
        size_scale=size_scale
    )
    
#-----------------------------------------------------------------------
def smart_imputation(dataframe):
    """Do column-wise imputation based on the dtypes."""
    
    # Load libraries
    import pandas as pd
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    
    # Getting names of columns based on data types
    categorical_features = list(dataframe.select_dtypes(include=['category']).columns)
    boolean_features = list(dataframe.select_dtypes(include=['bool']).columns)
    numerical_features = list(dataframe.select_dtypes(include='number').columns)
    datetime_features = list(dataframe.select_dtypes(include='datetime').columns)
    timedelta_features = list(dataframe.select_dtypes(include='timedelta').columns)
    object_features = list(dataframe.select_dtypes(include='object').columns)
    
    # Make a copy for proceeding imputation on
    df = dataframe.copy()
    
    #-----------------------------------------------------------------------
    # Imputation of numerical features
        
    # Proceed to imputation for ALL quantitative features
    # Do not work well on timeseries… (impute negative values)
    numerical_imputer = IterativeImputer()
    df[numerical_features] = numerical_imputer.fit_transform(df[numerical_features])
    
    #-----------------------------------------------------------------------
    # Imputation of timeseries features with a kNN regressor
    
    # Conversion of timeseries features to integers
    timeseries_features = datetime_features + timedelta_features
    df[timeseries_features] = df[datetime_features].values.astype('int64')
    
    # Standardization of the numerical features
    standardizer = StandardScaler()
    df_std = standardizer.fit_transform(df[numerical_features])
    
    # Train a kNN regressor model for each categorical feature
    for feature in timeseries_features:
        # filter the non-missing data
        mask = df[feature].notnull()
        # proceed imputation only if there is missing-data
        if not mask.all():
            # filter the data for training
            X_std = df_std[mask]
            y = df.loc[mask, feature]
            # train the model
            knn = KNeighborsRegressor(n_neighbors=5, n_jobs=-1).fit(X_std, y)
            # predict the missing data for missing data
            X_mis = df_std[~mask]
            df.loc[~mask, feature] = knn.predict(X_mis)

    #-----------------------------------------------------------------------
    # Imputation of categorical features with a kNN classifier
    
    # Standardization of the numerical features
    standardizer = StandardScaler()
    quantitative_features = numerical_features + timeseries_features
    df_std = standardizer.fit_transform(df[quantitative_features])
    
    # Train a kNN classifier model for each categorical feature
    for feature in categorical_features:
        # filter the non-missing data
        mask = df[feature].notnull()
        # proceed imputation only if there is missing-data
        if not mask.all():
            # filter the data for training
            X_std = df_std[mask]
            y = df.loc[mask, feature]
            # train the model
            knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1).fit(X_std, y)
            # predict the missing data for missing data
            X_mis = df_std[~mask]
            df.loc[~mask, feature] = knn.predict(X_mis)
    
    #------------------------------------------
    
    # Reverse dtypes for timeseries
    for feature in datetime_features:
        df[feature] = pd.to_datetime(df[feature])
        
    for feature in timedelta_features:
        df[feature] = pd.to_timedelta(df[feature]) 
   
    # Return the imputed dataframe
    return df

#-----------------------------------------------------------------------
def find_colinear_features(corrs, threshold=0.8, target_name='TARGET'):
    """Find colinear (highly correlated) features in a correlation DataFrame.
    This correlation DataFrame can be get from:
        - df.corr() for numerical features
        - dython.nominal.associations for categorical or mixed features
        
    Doesn't handle asymetry of Theil U coefficient yet.
    """
    
    # Initialize a dictionnary for colinear variables
    colinear_dict = {}

    # Masking target columns
    #-----------------------
    # In case a a single target column (or not specified)
    try:
        mask_target = (corrs.columns != target_name)
        
    # In case a several target columns
    except :
        mask_target = (~corrs.columns.isin(target_name))

        
    # Getting subsets of highly correlated features
    #----------------------------------------------
    # Initialize the list of substets
    list_of_correlated_subsets = []
    
    # For each column, record the variables that are above the threshold
    for feature in corrs:        
        # Select features with correlation coefficient higher than threshold
        mask1 = (corrs[feature].abs() > threshold)
        # Avoid selecting self feature (auto-correlation == 1)
        mask2 = (corrs.loc[feature].index != feature)
        # Final mask
        mask = mask1 & mask2
        # Create the subset of correlated features
        subset_of_correlated_features = set(corrs.index[mask])

        # If the subset of correlated features is not empty
        if subset_of_correlated_features:
            # Add the feature to the set
            subset_of_correlated_features.add(feature)
            # Add the values to the list of subsets, if not already
            if subset_of_correlated_features not in list_of_correlated_subsets:
                list_of_correlated_subsets.append(subset_of_correlated_features)

    # Checking for "transitivity" of correlation relationships
    #---------------------------------------------------------
    # for subset in list_of_correlated_subsets:
        # if 
    
    
    # Display and return results

    
    if not list_of_correlated_subsets:
        print("No found correlated features above threshold {}.".format(threshold))
    else:
        print("Subset(s) of correlated features above threshold {}:".format(threshold))
        for subset in list_of_correlated_subsets:
            print(subset)
    
    # Return results
    return list_of_correlated_subsets


#-----------------------------------------------------------------------
def PCA_features_reduction(X_std, var_threshold=0.9):
    """Return the principal components from PCA, until variance threshold."""

    from sklearn import decomposition
    
    # 
    print("Initial number of features:", X_std.shape[1])
    
    # Processing the PCA
    pca = decomposition.PCA()
    pca.fit(X_std)
    
    # Getting the explained variance ratio for each principal component
    scree = pca.explained_variance_ratio_

    # Getting the number of principal components to reach variance thresholds
    mask = scree.cumsum() > var_threshold
    nb_selected_features = len(scree[~mask]) + 1
    print("Number of selected features:", nb_selected_features)
    
    # Compute and displays the actual ratio of explained variance
    explained_variance_sum = scree.cumsum()[nb_selected_features-1]
    print("Cumulative explained variance:  {:.2f}%".format(explained_variance_sum*100))
    
    # Getting the projection of the data on the first components
    X_projected = pca.transform(X_std)[:,:nb_selected_features]
    
    return X_projected


#------------------------------------------------------------------------
def get_feature_names(columnTransformer):
    """This function returns features names from a 
    ColumnTransformer object, to keep track of featurse names."""

    output_features = []

    for transformers in columnTransformer.transformers:

        if transformers[0]!='remainder':
            pipeline = transformers[1]
            features = transformers[2]

            for i in pipeline:
                trans_features = []
                if hasattr(i,'categories_'):
                    for feature, categories in zip(features,i.categories_):
                        trans_features.extend(['{}_{}'.format(feature,cat)
                                                   for cat in categories])
                else:
                    trans_features = features
            output_features.extend(trans_features)

    return output_features


#----------------------------------------------------------------------
def define_preprocessor(dataframe, supervised=False):
    """This function returns a 'preprocessor' object wrapping
    different preprocessing column-wise steps.
        - impute missing values
        - target encode categorical features
        - standardization of all features
    """

    # Load required libraries and modules
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from category_encoders.target_encoder import TargetEncoder
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.preprocessing import FunctionTransformer
    import datetime
    
    # Warning if some columns have 'timedelta' or 'datetime' dtypes
    if not dataframe.select_dtypes(include=['timedelta', 'datetime']).empty:
        import warnings
        warnings.warn("Convert columns with 'timedelta' and 'datetime' dtypes before preprocessing.", UserWarning)
           
    # Warning if some columns' have object dtypes (not handled)
    if not dataframe.select_dtypes(include=['object']).empty:
        import warnings
        warnings.warn("Columns with 'object' dtypes are not handled and returned inchanged.", UserWarning)
        
   
    
    # Preprocessing pipeline for categorical features (supervised case)
    if supervised:
        categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), # simple imputation 
                ('target_encoder', TargetEncoder()), # target encoding
                ('scaler', StandardScaler()), # standardization after target encoding
                ])
    
    # Preprocessing pipeline for categorical features (unsupervised case)
    else:
        categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), # simple imputation 
                ('one_hot_encoder', LabelBinarizer()), # one_hot encoding for categorical nominal features
                ('scaler', StandardScaler()), # standardization after target encoding
                ])
    
    # Preprocessing pipeline for numeric features
    numeric_transformer = Pipeline(steps=[
            # ('imputer', IterativeImputer(max_iter=10)), # iterative imputation
            ('scaler', StandardScaler()), # standardization
             ])
    
    # Preprocessing pipeline for other dtypes features
    other_transformer = Pipeline(steps=[
            ('identity', FunctionTransformer(validate=False)), # identity function
             ])

    # Column-wise preprocessor using pipelines for numeric and categorical features
    preprocessor = ColumnTransformer(transformers=[
            # ('cat', categorical_transformer, list(dataframe.select_dtypes(include=['category', 'bool']).columns)),
            ('num', numeric_transformer, list(dataframe.select_dtypes(include='number', exclude=['datetime', 'timedelta']).columns)),
            # ('obj', other_transformer, list(dataframe.select_dtypes(include=['object', 'datetime', 'timedelta']).columns)),
            ])
    
    return preprocessor


#------------------------------------------------------------------------
def preprocessing(dataframe, target_name='TARGET', supervised=False):
    """Apply the preprocessing and get back columns name to
    returns a pd.DataFrame."""
    
    import pandas as pd
    
    #-------------------------------------------------------------------
    # Function to get the features names from a ColumnTransformer object
    def get_feature_names(columnTransformer):

        output_features = []

        for transformers in columnTransformer.transformers:

            if transformers[0]!='remainder':
                pipeline = transformers[1]
                features = transformers[2]

                for i in pipeline:
                    trans_features = []
                    if hasattr(i,'categories_'):
                        for feature, categories in zip(features,i.categories_):
                            trans_features.extend(['{}_{}'.format(feature,cat)
                                                       for cat in categories])
                    else:
                        trans_features = features
                output_features.extend(trans_features)

        return output_features
    
    #------------------------------------------------------------------
    # Function to revert a np.ndarray to pd.DataFrame with columns names
    def revert_to_df(X_preprocessed, preprocessor):
        features_name = get_feature_names(preprocessor)
        print(len(features_name))
        print(features_name)
        print(X_preprocessed.shape)
        X_preprocessed = pd.DataFrame(X_preprocessed, columns=features_name)
        return X_preprocessed
        
    #------------------------------------------------------------------
    X = dataframe.copy()
    
    # Specific operations for supervised case
    if supervised:
        # Dropping rows with missing target values
        X = X.dropna(subset=[target_name])

        # Setting the target variable (used for coloration only)
        y = X[target_name]

        # Removing the target from features
        X = X.drop(columns=[target_name])


    # Instanciate preprocessor
    preprocessor = define_preprocessor(X, supervised=supervised)
    
    # Apply the preprocessor to the data
    if supervised:
        X_preprocessed = preprocessor.fit_transform(X, y)
        X_preprocessed = revert_to_df(X_preprocessed, preprocessor)
        return (X_preprocessed, y_preprocessed)
    
    else:
        print("Shape of X:", X.shape)
        X_preprocessed = preprocessor.fit_transform(X)
        print("Shape of X_preprocessed:", X_preprocessed.shape)
        X_preprocessed = revert_to_df(X_preprocessed, preprocessor)
        return X_preprocessed
    
    
#-----------------------------------------------------------
def sampling_from_ndarray(array, size, replace=False):
    """Sample from np.ndarray. Do NOT work with pd.DataFrame"""
    idx = np.random.choice(array.shape[0], size, replace=replace)
    sample = array[idx,:]
    return sample


#-----------------------------------------------------------
def display_scree_plot(X_std):
    """This function displays the scree plot of proper values
    for PCA decomposition.
    
    Parameters
    --------
    X_std (pandas.DataFrame or np.ndarray):
        The standardised data (features).
   
    Future improvements
    -------------------
    * english translation
    * unit thicks
    """
    
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np
    
    pca = PCA()
    pca.fit(X_std)
    scree = pca.explained_variance_ratio_*100
    
    plt.figure(figsize=(20,10))
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres", fontsize=15)
    plt.show(block=False)

    
#-----------------------------------------------------------
def collapse_categories_by_frequency(categorical_Series, threshold=0.05, replacement='other'):
    """This function collapse together categories with frequencies
    (or relative frequencies) under the threshold. It returns a pd.Series
    with these categories replaced by 'replacement'."""

    # get frequencies of each modality
    categorical_Series.value_counts()

    # case where threshold is a frequency
    if threshold > 1:
        # define mask for modalities under threshold
        mask = (categorical_Series.value_counts() < threshold)

    # case where threshold is a relative frequency
    if threshold > 0 and threshold <1:
        # compute total frequency
        total_frequency = len(categorical_Series)
        # define mask for modalities under threshold
        mask = (categorical_Series.value_counts() < threshold*total_frequency)

    # apply mask and get categories names
    categories_to_collapse = list(categorical_Series.value_counts()[mask].index)

    # remap all 'categories to collapse' to the replacement value
    result = categorical_Series.apply(lambda x: replacement if x in categories_to_collapse else x)
    
    # displays informations
    print("Number of categorical modalities before collapsing: {}".format(categorical_Series.nunique()))
    print("Number of categorical modalities after collapsing: {}".format(result.nunique()))

    # return transformed pd.Series
    return result

#-----------------------------------------------------------------------
def IQR_outliers_mask(Series):
    """This function return a mask of outliers.
    Outliers are those with value less than (Q1 + 1.5*IQR) or bigger than (Q3 + 1.5*IQR)"""
    
    Q1 = Series.quantile(q=0.25, interpolation='linear')   # first quartile
    Q3 = Series.quantile(q=0.75, interpolation='linear')   # third quartile
    IQR = Q3 - Q1                                                      # inter-quartile range
    
    # definition of mask of outliers
    mask = (Series < Q1 - 1.5*IQR) | (Series > Q3 + 1.5*IQR)
    
    # application of mask
    outliers = Series[mask]
    
    # Displays number of outliers
    print("Number of detected outliers:", len(mask[mask]))
    
    # return the mask of outliers
    return mask

#-----------------------------------------------------------------
def categorical_distribution(dataframe, feature, ordinal=False):
    """Function plotting the bar-plot and pie-plot (as subplots) for 
    a distribution of categorical features."""
    
    # importing libraries
    import matplotlib.pyplot as plt

    # filtering non-null data for the feature
    mask = dataframe[feature].notnull()
    data_view = dataframe[mask]
    
    # Setting the data to plot
    x=data_view[feature]
    
    # Set frequencies and labels, sorting by index
    if ordinal==True:
        labels = list(x.value_counts().sort_index().index.astype(str))
        frequencies = x.value_counts().sort_index()
    else:
        labels = list(x.value_counts().sort_values(ascending=False).index.astype(str))
        frequencies = x.value_counts().sort_values(ascending=False)
    
    # Graphical properties of the main figure
    fig = plt.figure(figsize=(14, 6))
    
    plt.suptitle("Empiric statistical distribution: " + feature, fontsize=25)
    
    # Main graphical properties of the first subplot (histogram)
    ax1 = plt.subplot(121)
    ax1.set_xlabel("Values", fontsize=20)
    ax1.set_ylabel("Frequencies", fontsize=20)
    ax1.set_xticklabels(labels, rotation='45', horizontalalignment="right")

    # Main graphical properties of the second subplot (pieplot)
    ax2 = plt.subplot(122)
    ax2.set_xlabel("Relative frequencies", fontsize=20)
    
    # plotting the plots
    ax1.bar(labels, frequencies)
    ax2.pie(frequencies,
            autopct='%1.2f%%',
            shadow=True,
              )
    
    ax2.legend(labels)
    plt.show()
    return fig


#-------------------------------------------------------
def rows_fillingrate_histogram(dataframe):
    """This function plots an histogram of the distribution of the 
    filling rate for the rows of a dataframe."""

    # Import libraries
    import matplotlib.ticker as ticker
    import matplotlib.pyplot as plt
    import numpy as np

    # Count the non-null values on each row
    row_count = dataframe.count(axis=1)

    # Calculating filling rates
    nb_columns = dataframe.shape[1]
    filling_rates_row = row_count / nb_columns

    # Plotting histogramm
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.title("Rows' filling rate distribution", fontsize=25)
    plt.xlabel("Filling rate", fontsize=15)
    plt.ylabel("Frequency", fontsize=15)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    bins = np.linspace(0, 1, num=51)
    ax.hist(filling_rates_row, bins=bins)
    ax.xaxis.set_major_locator(plt.MaxNLocator(11))
    plt.show()
    
#-------------------------------------------------------
def fillingrate_filter_rows(dataframe, limit_rate):
    """This function drop the rows where the filling rate is less than a defined limit rate."""

    # Count of the values on each row
    rows_count = dataframe.count(axis=1)

    # Number of columns in the dataframe
    nb_columns = dataframe.shape[1]
    
    # Calculating filling rates
    filling_rates = rows_count / nb_columns

    # Define a mask of features with a filling_rate bigger than the limit rate
    mask = filling_rates > limit_rate
       
    # Get the number of rows under threshold
    number_rows_under_limit_rate = len(filling_rates[~mask])
    print("Number of rows with a filling rate below {:.2%}: {} rows.".format(limit_rate, number_rows_under_limit_rate))

    # Return a projection on the selection of features
    return dataframe[mask]

#-------------------------------------------------------
def empirical_distribution(dataframe, feature):
    """Function plotting the bar plot and a boxplot (as subplots) for a distribution."""
    
    # Loading libraries
    import matplotlib.pyplot as plt
    import seaborn as sns

    # filtering non-null data
    mask = dataframe[feature].notnull()
    data_view = dataframe[mask]
    
    # Setting the data to plot
    x = data_view[feature]
    
    # Create a figure instance, and the two subplots
    fig = plt.figure(figsize=(20, 10))
    plt.suptitle("Statistical distribution: " + feature, fontsize=25)
    ax1 = fig.add_subplot(211) # histogram
    ax2 = fig.add_subplot(212) # boxplot

    # Tell distplot to plot on ax1 with the ax argument
    sns.distplot(x, ax=ax1)
    ax1.set_ylabel("Frequency", fontsize=20)
    ax1.set_xlabel("")

    # Tell the boxplot to plot on ax2 with the ax argument
    medianprops = {'color':"black"}
    meanprops = {'marker':'o', 'markeredgecolor':'black', 'markerfacecolor':'firebrick'}
    sns.boxplot(x,
                ax=ax2,
                showfliers=True,
                medianprops=medianprops,
                showmeans=True,
                meanprops=meanprops)
    ax2.set_xlabel("Value", fontsize=20)
    return fig

#-------------------------------------------------------
def discrete_distribution(dataframe, feature):
    """Function plotting the bar plot and a boxplot (as subplots) for a distribution."""
    
    # Loading libraries
    import matplotlib.pyplot as plt
    import seaborn as sns

    # filtering non-null data
    mask = dataframe[feature].notnull()
    data_view = dataframe[mask]
    
    # Setting the data to plot
    x = data_view[feature]
    
    # Create a figure instance, and the two subplots
    fig = plt.figure(figsize=(20, 10))
    plt.suptitle("Statistical distribution: " + feature, fontsize=25)
    ax1 = fig.add_subplot(211) # histogram
    # ax2 = fig.add_subplot(312) # kde
    ax3 = fig.add_subplot(212) # boxplot

    # Tell distplot to plot on ax1 with the ax argument
    ax2 = ax1.twinx()
    sns.distplot(x, kde=False, ax=ax1)
    sns.distplot(x, hist=False, ax=ax2, kde_kws={'bw':1})
    
    ax1.set_ylabel("Frequency", fontsize=20)
    ax1.set_xlabel("")

    # Tell the boxplot to plot on ax2 with the ax argument
    medianprops = {'color':"black"}
    meanprops = {'marker':'o', 'markeredgecolor':'black', 'markerfacecolor':'firebrick'}
    sns.boxplot(x,
                ax=ax3,
                showfliers=True,
                medianprops=medianprops,
                showmeans=True,
                meanprops=meanprops)
    ax3.set_xlabel("Value", fontsize=20)
    return fig
