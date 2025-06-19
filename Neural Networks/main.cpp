#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <Eigen/Dense>
#include <sstream>
#include <string>

using namespace Eigen;
using namespace std;

void read_csv(string filename,vector<vector<double>>&data)
{
    ifstream file(filename);
    string line;
    while (getline(file,line))
    {
        stringstream ss(line);
        vector<double>row;
        string entry;
        while (getline(ss,entry,','))
        {
            row.push_back(stod(entry));
        }
        data.push_back(row);
    }
    
}

MatrixXd convert_to_matrix_format(vector<vector<double>>&data)
{
    int rows=data.size();
    int col=data[0].size();
    MatrixXd mat(rows,col);
    for(int i=0;i<rows;i++)
    {
        for(int j=0;j<col;j++)
        {
            mat(i,j)=data[i][j];
        }
    }
    return mat;
}

MatrixXd ReLU(const MatrixXd& Z)
{
    return Z.cwiseMax(0); //max (0,element) 
}

MatrixXd ReLU_derivative(const MatrixXd& Z)
{
    MatrixXd dz=Z.unaryExpr([](double z){ if(z>0) return 1.0; else return 0.0;});
    return dz;
}

VectorXd softmax(const VectorXd& z)
{
    VectorXd e_z=z.array().exp();
    return e_z/e_z.sum();
}

VectorXd forward_propagation(const VectorXd&x,const MatrixXd& W1,const VectorXd& b1,const MatrixXd& W2,const VectorXd& b2,const MatrixXd& W3,const VectorXd& b3,VectorXd& a1,VectorXd&a2,VectorXd& a3,VectorXd &z1,VectorXd &z2,VectorXd &z3)
{
    z1=W1*x+b1;
    a1=ReLU(z1);

    z2=W2*a1+b2;
    a2=ReLU(z2);

    z3=W3*a2+b3;
    a3=softmax(z3);

    VectorXd y_hat=a3;
    return y_hat;
}

double cross_entropy_loss(const VectorXd& y, const VectorXd& y_hat) {
    double epsilon = 1e-10;
    ArrayXd safe_hat = y_hat.array() + epsilon;
    return - (y.array() * safe_hat.log()).sum();
}


void backpropagation(const VectorXd& x,VectorXd& y,VectorXd& y_hat,VectorXd& a1,VectorXd& a2,MatrixXd& dW3,MatrixXd& dW2,MatrixXd& dW1,VectorXd& db1,VectorXd& db2,VectorXd& db3,const MatrixXd& W2,
    const MatrixXd& W3, const VectorXd& z1, const VectorXd& z2)
{
    VectorXd dz3=(y_hat-y);
    dW3=dz3*a2.transpose();
    db3=dz3;
    VectorXd dz2=(W3.transpose()*dz3).cwiseProduct(ReLU_derivative(z2));
    dW2=dz2*a1.transpose();
    db2=dz2;
    VectorXd dz1=(W2.transpose()*dz2).cwiseProduct(ReLU_derivative(z1));
    dW1=dz1*x.transpose();
    db1=dz1;

}

void mini_batch_gradient_descent(const MatrixXd& X_train,const MatrixXd& Y_train,MatrixXd& W1, VectorXd& b1,MatrixXd& W2, VectorXd& b2,MatrixXd& W3, VectorXd& b3)
{
    int n_samples=X_train.rows();
    int epochs=10;
    double alpha=0.01;
    for(int epoch=0;epoch<epochs;epoch++)
    {
        double epoch_loss=0.0;
        for(int i=0;i<n_samples;i++)
        {
            VectorXd x=X_train.row(i).transpose();
            VectorXd y=Y_train.row(i).transpose();
            VectorXd a1,a2,a3;
            VectorXd z1,z2,z3;
            forward_propagation(x,W1,b1,W2,b2,W3,b3,a1,a2,a3,z1,z2,z3);
            double loss=cross_entropy_loss(y,a3);
            epoch_loss+=loss;
            MatrixXd dW1, dW2, dW3;
            VectorXd db1, db2, db3;
            backpropagation(
                x, y, a3,
                a1, a2,
                dW3, dW2, dW1,
                db1, db2, db3,
                W2, W3, z1, z2
            );
            b3 -= alpha * db3;
            W2 -= alpha * dW2;
            W3 -= alpha * dW3;
            b2 -= alpha * db2;
            W1 -= alpha * dW1;
            b1 -= alpha * db1;
        }
        std::cout<< "Epoch " << (epoch + 1) << " Loss: " << epoch_loss / n_samples << endl;
    }
}


void evaluate_model(const MatrixXd &X_test, const MatrixXd &Y_test,
                    const MatrixXd &W1, const VectorXd &b1,
                    const MatrixXd &W2, const VectorXd &b2,
                    const MatrixXd &W3, const VectorXd &b3)
{
    int correct = 0;
    int n = X_test.rows();
    MatrixXi confusion = MatrixXi::Zero(10, 10);

    for (int i = 0; i < n; i++) {
        VectorXd x = X_test.row(i).transpose();
        VectorXd y_true = Y_test.row(i).transpose();

        VectorXd z1, z2, z3, a1, a2, a3;
        VectorXd y_hat = forward_propagation(x, W1, b1, W2, b2, W3, b3, z1, a1, z2, a2, z3, a3);

        int predicted = -1, actual = -1;
        y_hat.maxCoeff(&predicted);
        y_true.maxCoeff(&actual);

        if (predicted == actual)
            correct++;

        confusion(actual, predicted)++;
    }

    double accuracy = static_cast<double>(correct) / n;
    std::cout << "\nAccuracy: " << accuracy * 100 << "%" << endl;
    std::cout << "Confusion Matrix:\n" << confusion << endl;
}
int main()
{
    vector<vector<double>>x_train_data,y_train_data,x_test_data,y_test_data;
    read_csv("X_train.csv",x_train_data);
    read_csv("y_train.csv",y_train_data);
    read_csv("X_test.csv",x_test_data);
    read_csv("y_test.csv",y_test_data);
    MatrixXd X_train=convert_to_matrix_format(x_train_data);
    MatrixXd X_test=convert_to_matrix_format(x_test_data);
    MatrixXd Y_train=convert_to_matrix_format(y_train_data);
    MatrixXd Y_test=convert_to_matrix_format(y_test_data);

    MatrixXd W1=MatrixXd::Random(128,784)*sqrt(2.0/784);
    VectorXd b1=VectorXd::Zero(128);

    MatrixXd W2=MatrixXd::Random(64,128)*sqrt(2.0/128);
    VectorXd b2=VectorXd::Zero(64);

    int output_size=10;
    MatrixXd W3=MatrixXd::Random(10,64)*sqrt(2.0/64);
    VectorXd b3=VectorXd::Zero(10);
    mini_batch_gradient_descent(X_train, Y_train, W1, b1, W2, b2, W3, b3);
    evaluate_model(X_test, Y_test, W1, b1, W2, b2, W3, b3);
    return 0;
}