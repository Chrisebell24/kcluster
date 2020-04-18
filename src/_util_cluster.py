import numpy as np
import pandas as pd

def _test_pickle(x):
    import pickle
    pickle.dump(x, open('./data/test3.p',mode='wb'))
    assert False

def f_mean(x):
    return np.sum(x**2)

def f_median(x):
    return np.sum(abs(x))

def _fun_okc_2(params):
    datanorm, n_clusters, tolerance, max_iter, f1, f2, eval_measure = params
    '''
    Clustering function
    '''
        
    K = n_clusters
    datanorm = datanorm.copy()

    M, N = datanorm.shape
        
    
    S = (np.random.rand(M,K+1)>0.5)*1
    S[:,-1]=1
    
    W = np.random.rand(K+1,N)
    W[-1]=0
        
    sse=[]
    oprevse = np.exp(70)
    opercentse = 1
    i = 1

    while ( i <= max_iter ) & ( opercentse > tolerance ) :
        for k in range(0,K):
                
            minus_k =[h for h in list(range(K)) if h !=k]
                
            s_minus_k = S[:,minus_k]
            w_minus_k = W[minus_k]
                                
            dstar = datanorm-np.matmul(s_minus_k, w_minus_k)
                
            s = S[:,k]
            w = np.transpose(np.matrix(W[k]))
                
            prevse = np.exp(70)
            percentse = 1
            l = 1

            while ((l <= max_iter) & (percentse > tolerance)):
                
                w = dstar[s==1].aggregate(f1).values
                s = np.where(f2((dstar-w).T)<=f2((dstar).T), 1, 0)

                # re apply random weights if s is all 0s
                if sum(s)==0: s = (np.random.rand(len(s))>0.5)*1
                    
                se = f2(
                    ((dstar - np.matmul(
                        np.reshape(s, (len(s),1)), 
                        np.reshape(w, (1,len(w)))
                    ))).unstack()
                )
                
                percentse = 1 - se/prevse
                prevse = se
                
                l += 1
                
            S[:,k] = s
            W[k] = np.reshape(w, len(w))
            
        numer = f2((datanorm-np.matmul(S, W)).unstack())
        denom = f2(datanorm.unstack()-f1(datanorm.unstack()))
        
        sse.append(numer/denom)
            
        ose = numer
        opercentse = (oprevse - ose)/oprevse
        oprevse = ose
        i += 1
        
    if eval_measure == 'VAF':
        
        first = datanorm.unstack().values
        second = np.reshape(np.matmul(S,W), len(datanorm.unstack()), order='F')
        success_measure = pd.DataFrame(
            {
                'first':first,
                'second': second
            }
        ).reset_index(drop=True).corr().iloc[1,0]**2
        
    elif eval_measure == 'DAF':
        success_measure = 1 - sse[-1]
        
    else:
        raise ValueError('Did not recognize eval_measure: {}'.format(eval_measure))
            
    return {
        'success_measure': success_measure,
        'eval_measure': eval_measure,
        'groups': np.matmul(S[:,:K], 2**np.array(range(n_clusters))),
        'centroids': W[:K],
        'sse_percent': sse,
    }

def _choose_functions(f1, f2, lnorm):

    function_dict = {
        'mean': (np.mean, f_mean),
        'median': (np.median, f_median),
    }

    if f1 is None: f1 = function_dict[lnorm][0]
    if f2 is None: f2 = function_dict[lnorm][1]

    return f1, f2

def _choose_eval_measure(lnorm, eval_measure):
    '''
    logic to decide the determined success measure (range [0-1] )
    VAF - variance accounted for
    DAF - deviance accounted for
    '''
    if eval_measure == 'auto':
        
        if lnorm == 'mean':
            eval_measure = 'VAF'
        elif lnorm == 'median':
            eval_measure = 'DAF'
        elif lnorm == 'mode':
            eval_measure = 'MAF'
        else:
            eval_measure = 'VAF'
        
    else:
        eval_measure = eval_measure.upper()
            
    return eval_measure
    
def _apply_outlier_filter(X, outlier):
        # remove_outliers
    if outlier != None:
        for col in X.columns:
            X.loc[X[col].abs()>outlier, col] = np.nan
            
    return X

def _normalize(X):
        
    X_mean_ = np.mean(X)
    X_sd_ = np.std(X)
    X_normalized = (X-X_mean_)/X_sd_
        
    cols = set(X_normalized.columns)
    X_normalized.dropna(how='all', axis=1, inplace=True)
    new_cols = set(X_normalized.columns)
    diff_cols = cols.difference(new_cols)
        
    if len(diff_cols)>0:
        print('Dropped {} column(s) because of NaN values during normalization'.format(len(diff_cols)))
        print(diff_cols)
            
    rows = len(X_normalized)
    X_normalized.dropna(inplace=True)
    if len(X_normalized)<rows:
        print('Dropped {} row(s) because of NaNs'.format(rows-len(X_normalized)))
        
    return X_normalized