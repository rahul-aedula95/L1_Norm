from sklearn import linear_model
from sklearn import preprocessing
import numpy as np
from numpy import matrix
from numpy import linalg as LA




def reader(filename):

    data = np.loadtxt(filename)
    return data

def writer(mat,filename):

    np.savetxt(filename,mat)

def svdcalc(mat):

    u,s,v = np.linalg.svd(mat,full_matrices=False)

    return (u,s,v)


def svdrowmat(mat):

    mat_trans = np.transpose(mat)
    res = np.dot(mat_trans,mat)

    u,sigma,v = svdcalc(res)
    sigma = np.sqrt(sigma)
    return sigma[0]

def matrix_align(mat):
    row,col = np.shape(mat)

    if col == min(row,col):
        return mat

    return np.transpose(mat)    
        
def lasso(mat): 
        row,col = np.shape(mat)
        clf=linear_model.Lasso(alpha=0.01)
        clf.fit(mat,[i for i in xrange(0,row)]) 
        N=clf.coef_
        col1, = np.shape(N)
        N = np.reshape(N,(1,col1))

        return N

def norm(mat,axisval):

    return LA.norm(mat,axis=axisval,ord=1)


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


def normalizer(arr):
    arr = np.asarray(arr)
    arr = np.reshape(arr,(1,-1))
    res = preprocessing.normalize(arr, norm='l1')
    return res

def filter(arr):

    for i in xrange(0,len(arr)-1):
        arr[i] = arr[i] - arr[i+1]

    return arr        

def recursive(mat):

    row,col = np.shape(mat)
    arr = []
    z = np.zeros(shape=(row,1))
    for i in xrange(0,col):
        nrow,ncol = np.shape(mat)
        l1_row = lasso(mat)       
        ele = svdrowmat(l1_row)

        arr.append(ele) 

        row_norm = norm(l1_row,1)

        col_norm = norm(mat,0)

        #print row_norm
        #print col_norm
                
        col_del = find_nearest(col_norm,row_norm[0])

        new = np.delete(mat,[i for i in xrange(0,ncol) if i != col_del],axis=1)
                
        z = np.concatenate((z,new),axis=1)

        mat = np.delete(mat,[col_del],axis=1)
    z = np.delete(z,0,axis=1)

    print arr
    result = arr
    #result = filter(arr)
    print result
    result = normalizer(arr)

    return (result,z)
    



if __name__ == "__main__":

        file1 = "inputmat.txt"
        file2 = "output_singvals.txt"
        file3 = "output_ranked_matrix.txt"
        data1 = reader(file1)
        #data1 = matrix_align(data1)

        normalized_singvals,ranked_matrix = recursive(data1)
        writer(normalized_singvals,file2)
        ranked_matrix = np.transpose(ranked_matrix)
        writer(ranked_matrix,file3)
         

        