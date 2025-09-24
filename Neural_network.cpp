#include<bits/stdc++.h>
using namespace std;

random_device rd;
mt19937 gen(rd());

struct Neuron
{
    vector<double> weight;
    double bias;
    double output;
    double prev_output;
    double grad_loss_to_in;

    Neuron(){}

    Neuron(vector<double> inputs, string& activation_func, int fan_out) : bias(0.0), output(0.0) {
        for(auto& i:activation_func){
            i = tolower(i);
        }

        if (activation_func == "relu")
        {
            double stddev = sqrt(2.0/inputs.size());
            normal_distribution<> dis(0.0, stddev);
            weight.resize(inputs.size());
            for(auto& w : weight){
                w = dis(gen);
            }
        }else if (activation_func == "sigmoid")
        {
            double limit = sqrt(6.0/(inputs.size() + fan_out));
            uniform_real_distribution<> dis(-limit, limit);
            weight.resize(inputs.size());
            for(auto& w : weight){
                w = dis(gen);
            }
        }else
        {
            uniform_real_distribution<> dis(-1, 1);
            weight.resize(inputs.size());
            for(auto& w : weight){
                w = dis(gen);
            }
        }
    }

    void compute_output(const vector<double>& inputs, string& activation_func){
        prev_output = output;
        output = 0;
        for(auto i = 0; i < inputs.size(); i++){
            output += inputs[i]*weight[i];
        }
        output += bias;

        if (activation_func == "relu")
        {
            output = max(0.0, output);
        }else if (activation_func == "sigmoid")
        {
            output = 1.0 / (1.0 + exp(-output));
        }else
        {
            cout<<"Error in activation function string"<<endl;
        }
    }
};

vector<vector<Neuron>> init_network(vector<int> layer_count, vector<string>& activation_func, vector<double> inputs){
    vector<vector<Neuron>> result;
    for (int i = 0; i < layer_count.size(); i++)
    {
        vector<Neuron> tmp;
        bool is_end = i >= layer_count.size()-1;

        if (i == 0)
        {
            for (int j = 0; j < layer_count[i]; j++)
            {
                Neuron init;
                if (!is_end){
                    init = Neuron(inputs, activation_func[i], layer_count[i+1]);
                }else
                {
                    init = Neuron(inputs, activation_func[i], layer_count[i]);
                }
                init.compute_output(inputs, activation_func[i]);
                tmp.push_back(init);
            }
            result.push_back(tmp);
        }else if(i > 0){
            vector<double> prev_out;

            for (auto prev_neuron : result[i-1])
            {
                prev_out.push_back(prev_neuron.output);
            }
            
            for (int j = 0; j < layer_count[i]; j++)
            {
                Neuron init;
                if (!is_end){
                    init = Neuron(prev_out, activation_func[i], layer_count[i+1]);
                }else
                {
                    init = Neuron(prev_out, activation_func[i], layer_count[i]);
                }
                init.compute_output(prev_out, activation_func[i]);
                tmp.push_back(init);
            }
            result.push_back(tmp);
        }
    }
    return result;
}

void gradient_loss_to_input(vector<Neuron>& neuron_in_layer, vector<Neuron>& neuron_next_layer, string activation_func){
    if (activation_func == "sigmoid")
    {
        for (int i = 0; i < neuron_in_layer.size(); i++)
        {
            auto& neuron = neuron_in_layer[i];
            double sum_gradient = 0;
            for(auto& next_neuron : neuron_next_layer){
                sum_gradient += next_neuron.grad_loss_to_in * next_neuron.weight[i];
            }

            neuron.grad_loss_to_in = sum_gradient * neuron.output * (1-neuron.output);
        }
    }else if (activation_func == "relu")
    {
        for (int i = 0; i < neuron_in_layer.size(); i++)
        {
            auto& neuron = neuron_in_layer[i];
            double sum_gradient = 0;
            for(auto& next_neuron : neuron_next_layer){
                sum_gradient += next_neuron.grad_loss_to_in * next_neuron.weight[i];
            }
            double relu_deriv = (neuron.output > 0) ? 1 : 0;
            neuron.grad_loss_to_in = sum_gradient * relu_deriv;
        }
    }
}

void output_layer_gradient_loss_to_input(vector<Neuron>& neuron_in_layer, string activation_func, double label){
    for(auto& neuron : neuron_in_layer){
        double gradient;
        if (activation_func == "sigmoid")
        {
            gradient = neuron.output - label;
            neuron.grad_loss_to_in = gradient;
        }
    }
}

void backward_pass(vector<int> layer_count, vector<string> activation_func, vector<vector<Neuron>>& layer, double label){
    for (int i = layer.size() - 1; i > -1; i--)
    {
        if (i == layer.size() - 1)
        {
            output_layer_gradient_loss_to_input(layer[i], activation_func[i], label);
        }else
        {
            gradient_loss_to_input(layer[i], layer[i+1], activation_func[i]);
        }
    }
}

double binary_cross_entropy_loss(double label, double prediction){
    return -(label * log(prediction + 1e-15) + (1 - label) * log(1-prediction + 1e-15));
}

int main(int argc, char const *argv[])
{
    vector<double> data = {0.75, 0.6, 0.8, 1.0};
    vector<double> data2 = {0.45, 0.3, 0.21, 0.05};
    
    vector<vector<Neuron>> layer;
    vector<int> layer_count = {2, 3, 3, 1};
    vector<string> activation_func = {"Relu", "relu", "relu", "sigmoid"};
    
    layer = init_network(layer_count, activation_func, data);
    double prediction;
    for(auto i:layer[layer.size()-1]){
        prediction = i.output;
    }

    double loss1 = binary_cross_entropy_loss(1, prediction);
    backward_pass(layer_count, activation_func, layer, 1);
    cout<<"Prediction : "<<prediction<<endl;
    cout<<"Loss : "<<loss1<<endl;
    return 0;
}
