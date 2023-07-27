
# Testing methods.

from head import *



# Performs per-word voting.
def majVot(scrs, y_test):
    pred = np.empty(len(scrs), int)
    # For each image:
    for i in np.unique(y_test[:,0]):
        y_i = y_test[:,0] == i
        # For each word:
        for j in np.unique(y_test[y_i][:,1]):
            # Mask the characters of image i, and word j.
            mask = y_i & (y_test[:,1] == j)
            # Count the votes.
            if scrs.dtype == float:
                pred[mask] = np.argmax(scrs[mask].sum(0))
            else:
                v, c = np.unique(scrs[mask], return_counts=True)
                pred[mask] = v[c.argmax()]
    return pred



# Returns the sum of inverse distances for each font.  
def distScrs(knn, X):
    d, n = knn.kneighbors(X)
    I = np.identity(len(labels)) # One-hot.
    return (1 / (d[:,:,np.newaxis] + 1e-12) * I[knn._y[n]]).sum(1)



# Splits data into training and test sets according to fold index f.
# n is the number of folds.
def split(X_data, y_data, n, f):
    # Place every i'th (i != f) example in the training set.
    X_train = np.concatenate([np.concatenate( 
        [d for d in X_data[i::n]]) for i in range(n) if i != f])
    y_train = np.concatenate([np.concatenate(
        [d for d in y_data[i::n]]) for i in range(n) if i != f])
    # Place every f'th example in the test set.
    X_test = np.concatenate([d for d in X_data[f::n]])
    y_test = np.concatenate([d for d in y_data[f::n]])
    return X_train, X_test, y_train, y_test




# Perfroms cross-validation.
def cv(model, X_data, y_data, n=10, fPred=distScrs, fVote=majVot):
    scores = []
    # For each split:
    for f in range(n):
        # Split the data according to the current fold index.
        X_train, X_test, y_train, y_test = split(X_data, y_data, n, f)
        # Fit the model.
        model.fit(X_train, y_train[:,-1])
        # Generate the predictions.
        pred = fVote(fPred(model, X_test), y_test)
        # Store the accuracy score.
        scores.append(acc(y_test[:,-1], pred))
    return np.mean(scores), np.std(scores)


