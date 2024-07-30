#include <vector>
#include <cmath>

using namespace std;

class LogisticRegression {
private:
    vector<double> weights;

    double sigmoid(double z);

public:
    LogisticRegression(int numFeatures);

    void fit(const vector<vector<double>>& X_train, const vector<double>& y_train, double learningRate = 0.01, int epochs = 1000);
    double predictProbability(const vector<double>& features);
    int predict(const vector<double>& features, double threshold = 0.5);
};

// Member function definitions

LogisticRegression::LogisticRegression(int numFeatures) {
    // Initialize weights with zeros
    weights = vector<double>(numFeatures + 1, 0.0); // Additional weight for bias
}

double LogisticRegression::sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

void LogisticRegression::fit(const vector<vector<double>>& X_train, const vector<double>& y_train, double learningRate, int epochs) {
    int numFeatures = X_train[0].size();
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int i = 0; i < X_train.size(); ++i) {
            double prediction = predictProbability(X_train[i]);
            double error = y_train[i] - prediction;
            for (int j = 0; j < numFeatures; ++j) {
                weights[j] += learningRate * error * X_train[i][j];
            }
            // Update bias
            weights[numFeatures] += learningRate * error;
        }
    }
}

double LogisticRegression::predictProbability(const vector<double>& features) {
    double z = weights[features.size()];
    for (int i = 0; i < features.size(); ++i) {
        z += weights[i] * features[i];
    }
    return sigmoid(z);
}

int LogisticRegression::predict(const vector<double>& features, double threshold) {
    double probability = predictProbability(features);
    return (probability >= threshold) ? 1 : 0;
}