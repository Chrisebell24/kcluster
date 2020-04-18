import numpy as np
import pandas as pd
import multiprocessing as mp

# Cluster
from _util_cluster import _fun_okc_2, _choose_functions, _choose_eval_measure, _apply_outlier_filter, _normalize
# KCluster
from _util_kcluster import _init_random_clusters, _assign_clusters, _calculate_new_centroids, _cluster_calculation
def scree(
    X, 
    n_clusters,
    lnorm = 'mean',
    n_init = 10, 
    max_iter = 50,
    tolerance = 0.0001,
    f1=None,
    f2=None,
    eval_measure='auto',
    outlier = None,
    normalize=False,
    n_jobs=None
):
    
    results = []
    for k in n_clusters:
        model = Cluster()
        model.fit(
            X=X, 
            n_clusters=k,
            lnorm = lnorm,
            n_init = n_init, 
            max_iter = max_iter,
            tolerance = tolerance,
            f1=f1,
            f2=f2,
            eval_measure=eval_measure,
            outlier = outlier,
            normalize=normalize,
            n_jobs=n_jobs,
        )
        results.append(model)
    
    sdf = pd.DataFrame(
        [(m.n_clusters, m.success_measure_) for iid, m in enumerate(results)],
    columns=['Number of Clusters', 'Explained Variance']).set_index('Number of Clusters')
    
    return {'scree': sdf, 'models': results}


class KCluster:
    
    def __init__(
        self,
        n_clusters=8,
        n_init=10,
        max_iter=300,
        random_state=None,
        n_jobs=None,
    ):
        '''
        Parameters
        ----------
        n_clusters : int, optional, default: 8
            The number of clusters to form as well as the number of
            centroids to generate.
    
        n_init : int, default: 10
            Number of time the k-means algorithm will be run with different
            centroid seeds. The final results will be the best output of
            n_init consecutive runs in terms of inertia.
        max_iter : int, default: 300
            Maximum number of iterations of the k-means algorithm for a
            single run.
            
        random_state : int, RandomState instance or None (default)
            Determines random number generation for centroid initialization. Use
            an int to make the randomness deterministic.
            See :term:`Glossary <random_state>`.
            
        n_jobs : int, None
            number of jobs to run. Multiprocess if > 1.
            
        '''
        self.n_clusters = n_clusters
        self._random_state = random_state
        self._max_iter = max_iter
        self._n_init = n_init
        self._n_jobs = n_jobs

    def _normalize(self, X, method, alpha, manhattan_columns, current_measure_flag=False):

        '''
        Parameters
        ----------
        X : pandas dataframe
            Data to normalize
        method : str
            Method to use
        alpha : float
            numeric measure to use if applicable
        manhattan_columns : list
            list of median columns
        current_measure_flag : bool
            False: calculate new measures
            True: use stored measures

        Returns
        -------
        X : pandas dataframe
            Normalized X
        '''

        if current_measure_flag:
            center_measure = self._center_measure
            variation_measure = self._variation_measure
            measure_1 = self._measure_1
            measure_2 = self._measure_2

        else:
            center_measure, variation_measure, measure_1, measure_2 = {}, {}, {}, {}

        if method != None:

            for col, d in X.dtypes.items():
                if d in [float, int] and not all(X[col].isnull()):
                    if method == 'standard':

                        if not current_measure_flag:

                            if col not in manhattan_columns:
                                center_measure[col] = X[col].mean()
                                variation_measure[col] = X[col].std()

                            else:
                                center_measure[col] = X[col].median()
                                variation_measure[col] = abs(X[col] - center_measure[col]).mean()


                        X[col] = (X[col] - center_measure[col]) / variation_measure[col]


                    elif method == 'weibull':

                        if not current_measure_flag:
                            measure_1[col] = np.percentile(X[col].dropna(), 100 - 100 * alpha)
                            measure_2[col] = np.percentile(X[col].dropna(), 100 * alpha)

                        trimmed = np.maximum(np.minimum( X[col], measure_1[col]), measure_2[col])

                        if not current_measure_flag:
                            if col not in manhattan_columns:
                                center_measure[col] = trimmed.mean()
                                variation_measure[col] = trimmed.std()
                            else:
                                center_measure[col] = X[col].median()
                                variation_measure[col] = abs(X[col] - center_measure[col]).mean()

                        X[col] = (trimmed - center_measure[col]) / variation_measure[col]

        self._center_measure = center_measure
        self._variation_measure = variation_measure
        self._measure_1 = measure_1
        self._measure_2 = measure_2

        return X


    def fit(
        self, 
        X, 
        manhattan_list=[], 
        weights={}, 
        udf={}, 
        udf2={},
        udf3={},
        normalize_method=None,
        normalize_alpha=0.02,
    ):
        '''
        Parameters
        ----------
        X : pandas DataFrame
            DataFrame of data to cluster. The
            data types matter where str data types
            will be classified as categorical data
            even if the data is in numeric form
            
        manhattan_list : str/list
            List of columns in X that will use
            manhattan distance for classification.
            You should use this for interval/ordered
            numeric data. The default numeric distance
            measure will be eucledian distance
        
        weights : dict
            dictionary of weights. The sum of all the
            weights should be equal to one. By default,
            the weights will be evenly split up. If less
            than the number of columns are present in weights,
            then the remaining weight will be evenly split up
            among non-listed variables
        
        udf : dict
            User defined distance function
        
        udf2 : dict
            User defined centroid calculation function.
            This fictionary should have the column name
            in X with a function that takes the argument
            
        udf3 : dict
            'VAF' - mean, 'DAF' - median, 'MAF' - mode
            Dictionary of columns in df.columns that 
            correspond to other_columns that will
            have VAF, DAF, or MAF

        normalize_method : str/None

            standard : standard normal

            weibull : trim by percentile of normalize_alpha minimum and maximum
                        then apply standard normalization

        normalize_alpha : float
            if applicable alpha to use in normalization
        
        Example
        -------
        model = KCluster()
        model.fit(
            df, 
            manhattan_list=['Account Balance', 'Purpose'], 
            weights={'Creditability': 0.2}
        )
        '''
        
        weights = weights.copy()
        
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='ignore')
                
        self.X = X.copy()
        
        n_jobs = self._n_jobs
        if self._random_state != None:
            np.random.set_seed(self._random_state)
            
        if type(manhattan_list) is str: manhattan_list = [manhattan_list]
        
        n_clusters = self.n_clusters
        df = X.copy()
        
        eucledian_columns,manhattan_columns,mode_columns,other_columns = [],[],[],[]

        for col, data_type in zip(df.columns, df.dtypes):
            if col in manhattan_list:
                manhattan_columns.append(col)
            elif col in udf:
                other_columns.append(col)
            elif data_type in [float, int]:
                eucledian_columns.append(col)
            else:
                mode_columns.append(col)
        
        if weights != {}:
            
            avail_weight = 1.0 - sum(weights.values())
            if avail_weight < 1.0:
                non_issued_weight = avail_weight/(len(df.columns)-len(weights))
            else:
                non_issued_weight = 0.0
        else:
            avail_weight = 1.0
            non_issued_weight = avail_weight/df.shape[1]
         
        if non_issued_weight >0.0:
            for col in df.columns:
                if col not in weights:
                    weights[col] = non_issued_weight

        df = self._normalize(X=df, method=normalize_method, alpha=normalize_alpha, manhattan_columns=manhattan_columns)
        self._df = df
        self._normalize_method = normalize_method
        self._normalize_alpha = normalize_alpha
        
        arglist = [
            ((
                df,
                n_clusters,
                eucledian_columns,
                manhattan_columns,
                mode_columns,
                other_columns,
                udf,
                udf2,
                udf3,
                weights,
                self._max_iter,
            ))
            for i in range(self._n_init)
        ]
        if n_jobs ==1 or n_jobs == None:
            model_results = []
            for arg in arglist:
                model_results.append(_cluster_calculation(arg))
        else:
            pool = mp.Pool(min(mp.cpu_count()-1, n_jobs))
            model_results = pool.map(_cluster_calculation, arglist) 
            pool.close()
            pool.join()

        model_results = pd.DataFrame(
            model_results, 
            columns=['centroids', 'evaluation_measure', 'cluster_assignments'])
        
        model_results.dropna(inplace=True)
        
        if len(model_results)>0:

            best_model = model_results.loc[model_results['evaluation_measure'].fillna(0.0).idxmax()]

            self._model_results = model_results
            self.cluster_centers_ = best_model['centroids']
            self.eval_measure_ = best_model['evaluation_measure']
            self.labels_ = best_model['cluster_assignments'].values
            
        else:
            raise ValueError('No model results given - {} arglist'.format(len(arglist)))

        self._eucledian_columns, self._manhattan_columns, self._mode_columns, self._other_columns = eucledian_columns,manhattan_columns,mode_columns,other_columns
        self.weights_ = weights
        self._udf = udf

    def get_centers(self):

        centers = self.cluster_centers_
        for col in self._center_measure.keys():
            if col in centers.columns:
                if col in self._variation_measure and col in self._center_measure:
                    centers[col] = centers[col] * self._variation_measure[col] + self._center_measure[col]

        return centers

    def predict(self, X):

        X = self._normalize(X=X, method=self._normalize_method, alpha=self._normalize_alpha, manhattan_columns=self._manhattan_columns, current_measure_flag=True)

        return _assign_clusters(
            df=X,
            random_cluster_centers=self.cluster_centers_,
            udf=self._udf,
            weights=self.weights_,
            eucledian_columns=self._eucledian_columns,
            manhattan_columns=self._manhattan_columns,
            mode_columns=self._mode_columns,
            other_columns=self._other_columns
        )

class OverlappingCluster:       
    
    def fit(
        self, 
        X, 
        n_clusters,
        lnorm = 'mean',
        n_init = 10, 
        max_iter = 50,
        tolerance = 0.0001,
        f1=None,
        f2=None,
        eval_measure='auto',
        outlier = None,
        normalize=False,
        n_jobs=None,
    ):
        '''
        Parameters
        ----------
        X : pandas dataframe
            Numeric only pandas dataframe
            
        n_clusters : int
            number of clusters to fit
        
        lnorm : function
            function to minimize
            e.g. median, mean, ect...
            
        n_init : int, default: 10
            Number of time the k-means algorithm will be run with different
            centroid seeds. The final results will be the best output of
            n_init consecutive runs in terms of inertia.
        max_iter : int, default: 300
            Maximum number of iterations of the k-means algorithm for a
            single run.
            
        tolerance : float: 1e-4
            Relative tolerance with regards to inertia to declare convergence
            
        outlier : None/float
            remove from numeric if absolute value
            is greater than this number
            
        normalize : bool
            If True, standard normalize data
            
        n_jobs : int or None, optional (default=None)
            The number of jobs to use for the computation.
            
        Attributes
        ----------
        cluster_centers_ : array
            Coordinates of cluster centers
        labels_ :
            Labels of each point
        succcess_measure_ : float
            0 to 1 measure of explained variance compared to the 
            total variance. VAF is variance accounted for and is used for
            numeric, non-ordered data. DAF is deviance accounted for and is
            used for ordered data. MAF is matches accounted for and is used
            for categorical and categorical like data
        '''
        
        self._outlier = outlier
        self.max_iter = max_iter
        self.tolerance = tolerance
        
        X = _apply_outlier_filter(X, outlier)
        self.X = X
        if n_jobs == None: n_jobs = 1
            
        if normalize:
            self.X_normalized = _normalize(X)
        else:
            self.X_normalized = self.X
        
        self.n_clusters = n_clusters
        
        f1, f2 = _choose_functions(f1, f2, lnorm)
        eval_measure = _choose_eval_measure(lnorm, eval_measure)

        assert type(X) is pd.DataFrame, 'data is not a pandas dataframe. Type: {}'.format(type(X))
        

        arglist = [
            ((X, n_clusters, tolerance, max_iter, f1, f2, eval_measure))
            for i in range(1, n_init)
        ]
        if n_jobs ==1 or n_jobs == None:
            results = []
            for arg in arglist:
                results.append(_fun_okc_2(arg))
        else:
            pool = mp.Pool(min(mp.cpu_count()-1, n_jobs))
            results = pool.map(_fun_okc_2, arglist) 
            pool.close()
            pool.join()
        
        # post processing
        prevsse = 100
        for z in results:
            zsse_percent = pd.Series(z['sse_percent'])
            zsse_percent = zsse_percent[zsse_percent>0]
            zsse_percent = zsse_percent.iloc[-1] # take most recent sse that's positive
            
            if zsse_percent < prevsse:

                prevsse = zsse_percent
                self.results_ = z
                
        self.labels_ = self.results_['groups']
        self.cluster_centers_ = self.results_['centroids']
        self.success_measure_ = self.results_['success_measure']
