import numpy as np
import pandas as pd
import multiprocessing as mp
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

def _cluster_calculation(params):
    '''
    Run a KVariable calculation
    
    Parameters
    ----------
    df : pandas dataframe
        dataframe of data
    
    n_clusters : int
        number of clusters
    
    eucledian_columns : list
        Columns in df to use eucledian distance
        
    manhattan_columns : list
        Columns in df to use manhattan distance
        
    mode_columns : list
        Columns in df to use mode distance calcualtion
        
    other_columns : list
        Columns to use user defined function
        
    udf : dict
        dictionary of user defined functions
        that correspond to df columns. This is
        the function that will calculate a distance
        measure
        
    udf2 : dict
        Dictionary of columns in df.columns that 
        correspond to other_columns that will
        calculate the center of a centroid
    
    udf3 : dict
        'VAF' - mean, 'DAF' - median, 'MAF' - mode
        Dictionary of columns in df.columns that 
        correspond to other_columns that will
        have VAF, DAF, or MAF
    
    weights : dict
        weights corresponding to df columns
    
    max_iter : int
        maximum iterations
        
    Returns
    -------
    new_centroids : pandas dataframe
        n_clusters*len(df.columns) cluster
        
    evaluation_measure : float
        measure 0 - 1 for explained variance
        by cluster
        
    cluster_assignments : pandas series
        Pandas series of cluster assignments
    '''
    df,n_clusters,eucledian_columns, manhattan_columns, mode_columns, other_columns,udf, udf2, udf3, weights, max_iter = params
    ### Step 1: Select k points at random as cluster centers

    tolerance = 0.001
    
    random_cluster_centers = _init_random_clusters(n_clusters=n_clusters, df=df)
    old_cluster_assignments = pd.Series([0]*len(df))

    n_iterations = 0
    bypass = False
    isOptimal = False
    old_distance = 10e10
    first=True
    break_out = False
    
    while n_iterations<=max_iter:
        
        n_iterations+=1
        
        ### Step 2: Assign objects to their closest cluster center according to the Euclidean distance function
        cluster_assignments, distance = _assign_clusters(
            df=df, 
            random_cluster_centers=random_cluster_centers, 
            udf=udf, 
            weights=weights,
            eucledian_columns=eucledian_columns,
            manhattan_columns=manhattan_columns,
            mode_columns=mode_columns,
            other_columns=other_columns,
        )
        
        if (1 - distance / old_distance) < tolerance and not first:
            # change in distance measure is less than 
            break_out = True
        else:
            first = False


        if not False in (cluster_assignments==old_cluster_assignments).unique() and n_iterations>1 and (bypass or isOptimal): break

        ### Step 3: Calculate the centroid or mean of all objects in each cluster.
        new_centroids = _calculate_new_centroids(
            df=df,
            cluster_assignments=cluster_assignments,
            eucledian_columns=eucledian_columns,
            manhattan_columns=manhattan_columns,
            mode_columns=mode_columns,
            other_columns=other_columns,
            udf=udf2,
        )
        
        number_of_missing_clusters = n_clusters-new_centroids.shape[0]
        if number_of_missing_clusters>0:
            # dropped 1 or more clusters - place more clusters
            bypass = False
            missing_centroids = _init_random_clusters(n_clusters=number_of_missing_clusters, df=df)
            new_centroids = new_centroids.append(missing_centroids, ignore_index=True)
        else:
            bypass = True
            
        new_centroids.index.name='cluster'
        
        old_cluster_assignments = cluster_assignments
        
        old_distance = distance
        random_cluster_centers = new_centroids
        
        if break_out:
            break
        
    mapped_cluster = new_centroids.loc[cluster_assignments]
    
    X = df
    evaluation_measure = 0
    for col in X.columns:
        if col in eucledian_columns or (col in udf3 and udf3[col].upper()=='VAF'):
            # eval measure
            numer = ((X[col]-X[col].mean()).values*(mapped_cluster[col]-mapped_cluster[col].mean()).values).sum()
            denom = np.sqrt(np.power(X[col]-X[col].mean(), 2).sum()*np.power(mapped_cluster[col]-mapped_cluster[col].mean(), 2).sum())
            evaluation_measure += weights[col]*numer/denom
        
        elif col in manhattan_columns or (col in udf3 and udf3[col].upper()=='DAF'):
            # manhattan measure
            numer = sum((X[col]-X[col].mean()).values*(mapped_cluster[col]-mapped_cluster[col].median()).values)
            denom = (X[col]-X[col].mean()).abs().sum()*(mapped_cluster[col]-mapped_cluster[col].median()).abs().sum()
            evaluation_measure += weights[col]*numer/denom
        
        elif col in mode_columns or (col in udf3 and udf3[col].upper()=='MAF'):
            # mode measure
            evaluation_measure += weights[col]*sum((X[col].values == mapped_cluster[col].values)*1)/len(X)
            
        else:
            raise ValueError('Do not recognize column measure. Maybe provide measure for {} in udf3'.format(col))
    
    return new_centroids, evaluation_measure, cluster_assignments
    

def _calculate_new_centroids(
    df, 
    cluster_assignments,
    eucledian_columns,
    manhattan_columns,
    mode_columns,
    other_columns,
    udf,
):
    '''
    Parameters
    ----------
    df : pandas dataframe
        dataframe of data
    cluster_assignments : pandas series
        series of integers that assign to
        numbered clusters
    eucledian_columns : list
        Columns in df to use eucledian distance
    manhattan_columns : list
        Columns in df to use manhattan distance
    mode_columns : list
        Columns in df to use mode distance calcualtion
    other_columns : list
        Columns to use user defined function
    udf : dict
        Dictionary of columns in df.columns that 
        correspond to other_columns that will
        calculate the center of a centroid
    
    Returns
    -------
    pandas dataframe of new clusters
    '''
    df = df.copy().reset_index(drop=True)
    
    new_centroids = []
    for k in sorted(cluster_assignments.unique()):
        dfslice = df[cluster_assignments==k]

        centroid = []
        for col in dfslice.columns:

            if col in eucledian_columns:
                center_value = dfslice[col].mean()

            elif col in manhattan_columns:
                center_value = dfslice[col].median()

            elif col in mode_columns:
                center_value = dfslice[col].mode()
                if len(center_value)>0: center_value = center_value.iloc[0]
                
            else:
                center_value = dfslice[col].apply(udf[col])

            centroid.append(center_value)

        new_centroids.append(tuple(centroid))

    new_centroids = pd.DataFrame(new_centroids, columns=df.columns)
    return new_centroids

def _assign_clusters(
    df, 
    random_cluster_centers, 
    udf, 
    weights,
    eucledian_columns,
    manhattan_columns,
    mode_columns,
    other_columns
):
    '''
    Parameters
    ----------
    df : pandas dataframe
        data of pandas dataframe
    random_cluster_centers : pandas dataframe
        pandas dataframe of clusters
    udf : dict
        dictionary of user defined functions
        that correspond to df columns
    weights : dict
        weights corresponding to df columns
    eucledian_columns : list
        Columns in df to use eucledian distance
    manhattan_columns : list
        Columns in df to use manhattan distance
    mode_columns : list
        Columns in df to use mode distance calcualtion
    other_columns : list
        Columns to use user defined function
    
    Returns
    -------
    pandas series of cluster assignments
    distance : float
        distance measure for all clusters
    '''
    distance_list = []
    for cluster_number, cluster in random_cluster_centers.iterrows():

        distance = pd.Series(np.zeros_like(df.iloc[:,0]))

        if eucledian_columns != []:
            for c in eucledian_columns:
                distance += weights[c]*euclidean_distances(df[[c]], pd.DataFrame([cluster[[c]]])).ravel()

        if manhattan_distances != []:
            for c in manhattan_columns:
                distance += weights[c]*manhattan_distances(df[[c]], pd.DataFrame([cluster[[c]]])).ravel()

        if mode_columns != []:
            for c in mode_columns:
                d = [weights,df,cluster,c, distance]
                
                try:
                    distance += weights[c]*(df[[c]]!=cluster[[c]]).reset_index(drop=True)[c]*1
                except:
                    import pickle
                    pickle.dump(d, open('./data/mode_cluster.p', 'wb'))
                    raise ValueError('./data/mode_cluster.p')

        if other_columns != []:
            for c in other_columns:
                user_defined_distance_function = udf[c]
                distance += weights[c]*user_defined_distance_function(df[[c]], cluster[[c]])

        distance_list.append(distance)

    # M*n_cluster dataframe of distances
    distance_df = pd.concat(distance_list, axis=1).astype(float)
    cluster_assignment = distance_df.idxmin(axis=1)
    
    distance = distance_df.min(axis=1).sum()
    
    return cluster_assignment, distance

def _init_random_clusters(n_clusters, df):
    '''
    Parameters
    ----------
    n_clusters : int
        Number of clusters you want to create
    df : pandas DataFrame
        Data - The data type is import for how 
        random values are made. If a numeric variable
        is not in a numeric data type, it will treat
        the column as a categorical
    Returns
    -------
    random_cluster_centers : pandas DataFrame
        A number_of_clusters*len(df.columns) dataframe
        with random starting points for each cluster
    '''
    random_cluster_centers = []
    for k in range(n_clusters):
        random_cluster_centers.append([])
        for col, data_type in zip(df.columns, df.dtypes):
            if data_type in [int, float]:
                # numeric data type
                random_value = (df[col].max()-df[col].min())*np.random.rand()

            else:
                # categorical data type
                random_value = np.random.choice(df[col])

            random_cluster_centers[-1].append(random_value)

    random_cluster_centers = pd.DataFrame(random_cluster_centers, columns=df.columns)
    random_cluster_centers.index.name='cluster'

    return random_cluster_centers
