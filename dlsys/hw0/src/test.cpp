#include <cmath>
#include <iostream>
using namespace std;

// namespace py = pybind11;

float* matrixProduct(const float* X, const float* Y, size_t m, size_t n, size_t k){
    /*
    X -> (m, n)
    Y -> (n, k)
    */
   float* outputMatrix = (float*) malloc(sizeof(float) * m * k);
   memset(outputMatrix, 0, sizeof(float) * m * k);
    for(size_t i=0; i < m; i++){
        for(size_t j=0; j < k; j++){
            for(size_t z=0; z < n; z++){
                outputMatrix[i * k + j] += X[i * n + z] * Y[z * k +j]; 
            }
        }
    }
    return outputMatrix;
}

float* transpose(const float* X, size_t m, size_t n){
    float* outputMatrix = (float*) malloc(sizeof(float) * m * n);
    memset(outputMatrix, 0, sizeof(float) * m * n);
    for(size_t i=0; i<n; i++){
        for(size_t j=0; j<m; j++){
            outputMatrix[i * m + j] = X[j * n + i];
        }
    }
    return outputMatrix;
}

float* softmax(const float* X, const float* theta, size_t m, size_t n, size_t k){
    float* norm_X = (float*) malloc(sizeof(float) * m * k);
    memset(norm_X, 0, sizeof(float) * m * k);
    for(size_t i=0; i<m; i++){
        float sum = 0.0;
        for(size_t j=0; j<k; j++){
            for(size_t z=0; z<n; z++){
                norm_X[i * k + j] += X[i * n + z] * theta[z * k + j];
            }
            norm_X[i * k + j] = exp(norm_X[i * k + j]);
            sum += norm_X[i * k + j];
        }
        for(size_t j=0; j<k; j++){
            norm_X[i * k + j] = norm_X[i*k + j] / sum;
        }
    }
    return norm_X;
}

void printMatrix(const float* M, size_t m, size_t n){
    for(size_t i = 0; i<m; i++){
        for(size_t j=0; j<n; j++){
            cout << M[i*n +j] << " ";
        }
        cout << endl;
    }
}
void softmax_regression_batch(const float* X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr)
{
    float* Z = softmax(X, theta, m, n, k);
    // cout << "Z" << endl;
    // printMatrix(Z, m, k);
    for(size_t i=0; i<m; i++){
        Z[i * k + (uint8_t)y[i]] -= 1.0;
    }

    // float* res = (float*) malloc(sizeof(float) * m * k);
    float* X_t = transpose(X, m, n);
    float* grad_loss = matrixProduct(X_t, Z, n, m, k);
    // cout << "grad: " << endl;
    // printMatrix(grad_loss, n, k);
    for(size_t i=0; i<n; i++){
        for(size_t j=0; j<k; j++){
            theta[i*k + j] -= grad_loss[i*k + j] / (float)m * lr ;
            // cout << grad_loss[i+n + j] << " ";
        }
        // cout << endl;
    }
    free(Z);
    free(X_t);
    free(grad_loss);
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    // size_t per_example_size = sizeof(float) * n;
    size_t remain_example = m % batch;
    size_t batch_num = m / batch;
    const float* X_start_ptr = X ;
    const unsigned char* y_start_ptr = y;
    for(size_t i = 0; i < batch_num; i++){
        X_start_ptr = X + n * batch * i;
        y_start_ptr = y + batch * i;
        softmax_regression_batch(X_start_ptr, y_start_ptr, theta, batch, n, k, lr);
    }
    if(remain_example)
        softmax_regression_batch(X_start_ptr, y_start_ptr, theta, remain_example, n, k, lr);
    /// END YOUR CODE
}

int main(){
    float array_1[] = {1, 2, 3, 1, 1, 1};
    float array_2[] = {1, 0, 0, 1, 1, 1};
    unsigned char y[] = {0, 2, 1}; 
    float lr = 0.1;
    size_t batch = 1;
    size_t m = 3;
    size_t n = 2; 
    size_t k = 3;
    softmax_regression_epoch_cpp(array_1, y, array_2, m, n, k, lr, batch);
    float* productRes = matrixProduct(array_1, array_2, m, n, k);
    float* transposeRes = transpose(array_1, 3, 2);
    float* softmaxRes = softmax(array_1, array_2, m, n, k);
    // for(size_t i=0; i<n; i++){
    //     for(size_t j=0; j<k; j++){
    //         cout << array_2[i*k + j] << " ";
    //     }
    //     cout << endl;
    // }
    cout << "theta" << endl;
    printMatrix(array_2, n, k);
    // for(int i=0; i<m*k; i++){
    //     cout << softmaxRes[i] << " ";
    // }
    // cout << endl;
    // for(int i=0; i<m*k; i++){
    //     cout << productRes[i] << " ";
    // }
    // cout << endl;   
    // for(int i=0; i<m*n; i++){
    //     cout << transposeRes[i] << " ";
    // }
}