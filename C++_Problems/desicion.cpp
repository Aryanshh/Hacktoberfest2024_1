#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

using namespace std;

// Define a structure for each node in the decision tree
struct Node {
    int feature_index;        // Index of the feature for splitting
    double threshold;         // Threshold value for splitting
    Node* left;               // Left child node
    Node* right;              // Right child node
    int class_label;          // Class label for leaf nodes
    bool is_leaf;             // Is the node a leaf node?

    Node() : left(NULL), right(NULL), is_leaf(false), class_label(-1) {}
};

// Function to calculate Gini Index for a split
double giniIndex(vector<vector<double>>& dataset, vector<int>& classes, int feature_index, double threshold) {
    vector<int> left_count(2, 0);
    vector<int> right_count(2, 0);

    for (int i = 0; i < dataset.size(); i++) {
        if (dataset[i][feature_index] < threshold)
            left_count[classes[i]]++;
        else
            right_count[classes[i]]++;
    }

    int left_size = left_count[0] + left_count[1];
    int right_size = right_count[0] + right_count[1];
    int total_size = left_size + right_size;

    double left_gini = 1.0;
    if (left_size > 0)
        left_gini -= pow((double)left_count[0] / left_size, 2) + pow((double)left_count[1] / left_size, 2);

    double right_gini = 1.0;
    if (right_size > 0)
        right_gini -= pow((double)right_count[0] / right_size, 2) + pow((double)right_count[1] / right_size, 2);

    return ((double)left_size / total_size) * left_gini + ((double)right_size / total_size) * right_gini;
}

// Function to get the best split for the dataset
pair<int, double> getBestSplit(vector<vector<double>>& dataset, vector<int>& classes) {
    int best_feature = -1;
    double best_threshold = 0;
    double best_gini = numeric_limits<double>::max();

    for (int feature_index = 0; feature_index < dataset[0].size(); feature_index++) {
        for (int i = 0; i < dataset.size(); i++) {
            double threshold = dataset[i][feature_index];
            double gini = giniIndex(dataset, classes, feature_index, threshold);

            if (gini < best_gini) {
                best_gini = gini;
                best_feature = feature_index;
                best_threshold = threshold;
            }
        }
    }

    return {best_feature, best_threshold};
}

// Recursive function to build the decision tree
Node* buildTree(vector<vector<double>>& dataset, vector<int>& classes, int depth, int max_depth) {
    // Stop if we have reached max depth or if there's only one class left
    int class_0 = count(classes.begin(), classes.end(), 0);
    int class_1 = classes.size() - class_0;

    if (class_0 == 0 || class_1 == 0 || depth >= max_depth) {
        Node* leaf = new Node();
        leaf->is_leaf = true;
        leaf->class_label = (class_0 > class_1) ? 0 : 1;
        return leaf;
    }

    // Find the best split
    pair<int, double> best_split = getBestSplit(dataset, classes);

    // Split dataset based on the best split
    vector<vector<double>> left_data, right_data;
    vector<int> left_classes, right_classes;

    for (int i = 0; i < dataset.size(); i++) {
        if (dataset[i][best_split.first] < best_split.second) {
            left_data.push_back(dataset[i]);
            left_classes.push_back(classes[i]);
        } else {
            right_data.push_back(dataset[i]);
            right_classes.push_back(classes[i]);
        }
    }

    // Create the current node
    Node* node = new Node();
    node->feature_index = best_split.first;
    node->threshold = best_split.second;

    // Recursively build left and right children
    node->left = buildTree(left_data, left_classes, depth + 1, max_depth);
    node->right = buildTree(right_data, right_classes, depth + 1, max_depth);

    return node;
}

// Function to make predictions using the decision tree
int predict(Node* tree, vector<double>& sample) {
    if (tree->is_leaf)
        return tree->class_label;

    if (sample[tree->feature_index] < tree->threshold)
        return predict(tree->left, sample);
    else
        return predict(tree->right, sample);
}

int main() {
    // Define a small dataset with two features
    vector<vector<double>> dataset = {
        {2.3, 4.5},
        {1.5, 3.1},
        {3.8, 2.7},
        {2.9, 5.2},
        {4.1, 1.9}
    };

    // Define class labels (0 or 1)
    vector<int> classes = {0, 0, 1, 1, 1};

    // Set maximum depth for the tree
    int max_depth = 3;

    // Build the decision tree
    Node* tree = buildTree(dataset, classes, 0, max_depth);

    // Test the decision tree with a new sample
    vector<double> test_sample = {3.0, 4.0};
    int predicted_class = predict(tree, test_sample);

    cout << "Predicted class: " << predicted_class << endl;

    return 0;
}
