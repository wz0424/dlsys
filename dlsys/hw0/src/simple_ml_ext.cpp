#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

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

// void printMatrix(const float* M, size_t m, size_t n){
//     for(size_t i = 0; i<m; i++){
//         for(size_t j=0; j<n; j++){
//             cout << M[i*n +j] << " ";
//         }
//         cout << endl;
//     }
// }
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


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
