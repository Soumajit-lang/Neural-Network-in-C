#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUTS 2
#define HIDDEN 2
#define OUTPUTS 1
#define EPOCHS 10000
#define LEARNING_RATE 0.1

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of sigmoid
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// Generate random weight
double random_weight() {
    return ((double)rand() / RAND_MAX) * 2 - 1;
}

int main() {

    srand(time(0));

    // XOR Dataset
    double inputs[4][2] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    double targets[4][1] = {
        {0},
        {1},
        {1},
        {0}
    };

    // Weights
    double weights_input_hidden[INPUTS][HIDDEN];
    double weights_hidden_output[HIDDEN][OUTPUTS];

    // Biases
    double bias_hidden[HIDDEN];
    double bias_output[OUTPUTS];

    // Initialize weights randomly
    for(int i = 0; i < INPUTS; i++) {
        for(int j = 0; j < HIDDEN; j++) {
            weights_input_hidden[i][j] = random_weight();
        }
    }

    for(int i = 0; i < HIDDEN; i++) {
        for(int j = 0; j < OUTPUTS; j++) {
            weights_hidden_output[i][j] = random_weight();
        }
    }

    for(int i = 0; i < HIDDEN; i++)
        bias_hidden[i] = random_weight();

    for(int i = 0; i < OUTPUTS; i++)
        bias_output[i] = random_weight();

    // Training
    for(int epoch = 0; epoch < EPOCHS; epoch++) {

        for(int sample = 0; sample < 4; sample++) {

            // Hidden layer
            double hidden[HIDDEN];

            for(int i = 0; i < HIDDEN; i++) {
                hidden[i] = bias_hidden[i];

                for(int j = 0; j < INPUTS; j++) {
                    hidden[i] += inputs[sample][j] * weights_input_hidden[j][i];
                }

                hidden[i] = sigmoid(hidden[i]);
            }

            // Output layer
            double output[OUTPUTS];

            for(int i = 0; i < OUTPUTS; i++) {
                output[i] = bias_output[i];

                for(int j = 0; j < HIDDEN; j++) {
                    output[i] += hidden[j] * weights_hidden_output[j][i];
                }

                output[i] = sigmoid(output[i]);
            }

            // Calculate output error
            double output_error[OUTPUTS];

            for(int i = 0; i < OUTPUTS; i++) {
                output_error[i] =
                    (targets[sample][i] - output[i]) *
                    sigmoid_derivative(output[i]);
            }

            // Hidden layer error
            double hidden_error[HIDDEN];

            for(int i = 0; i < HIDDEN; i++) {

                hidden_error[i] = 0;

                for(int j = 0; j < OUTPUTS; j++) {
                    hidden_error[i] +=
                        output_error[j] * weights_hidden_output[i][j];
                }

                hidden_error[i] *= sigmoid_derivative(hidden[i]);
            }

            // Update hidden-output weights
            for(int i = 0; i < HIDDEN; i++) {
                for(int j = 0; j < OUTPUTS; j++) {
                    weights_hidden_output[i][j] +=
                        LEARNING_RATE * output_error[j] * hidden[i];
                }
            }

            // Update input-hidden weights
            for(int i = 0; i < INPUTS; i++) {
                for(int j = 0; j < HIDDEN; j++) {
                    weights_input_hidden[i][j] +=
                        LEARNING_RATE * hidden_error[j] * inputs[sample][i];
                }
            }

            // Update biases
            for(int i = 0; i < HIDDEN; i++)
                bias_hidden[i] += LEARNING_RATE * hidden_error[i];

            for(int i = 0; i < OUTPUTS; i++)
                bias_output[i] += LEARNING_RATE * output_error[i];
        }
    }

    // Testing
    printf("Trained Neural Network Output:\n");

    for(int sample = 0; sample < 4; sample++) {

        double hidden[HIDDEN];

        for(int i = 0; i < HIDDEN; i++) {
            hidden[i] = bias_hidden[i];

            for(int j = 0; j < INPUTS; j++) {
                hidden[i] += inputs[sample][j] * weights_input_hidden[j][i];
            }

            hidden[i] = sigmoid(hidden[i]);
        }

        double output = bias_output[0];

        for(int i = 0; i < HIDDEN; i++) {
            output += hidden[i] * weights_hidden_output[i][0];
        }

        output = sigmoid(output);

        printf("%.0f XOR %.0f = %.4f\n",
               inputs[sample][0],
               inputs[sample][1],
               output);
    }

    return 0;
}