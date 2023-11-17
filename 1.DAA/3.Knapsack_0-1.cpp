#include<bits/stdc++.h>
using namespace std;

int f(int ind, int W, vector<int> &wt, vector<int> &val, vector<vector<int>> &dp) {
    if (ind == 0) {
        if (wt[0] <= W) return val[0];
        return 0;
    }

    if (dp[ind][W] != -1) return dp[ind][W];

    int notTake = 0 + f(ind - 1, W, wt, val, dp);
    int take = INT_MIN;
    if (W >= wt[ind]) take = val[ind] + f(ind - 1, W - wt[ind], wt, val, dp);

    return dp[ind][W] = max(take, notTake);
}

int tabulatedKnapsack(int n, int W, vector<int> wt, vector<int> val) {
    vector<vector<int>> dp(n, vector<int>(W + 1, 0));

    for (int i = wt[0]; i <= W; i++) dp[0][i] = val[0];

    for (int ind = 1; ind < n; ind++) {
        for (int w = 0; w <= W; w++) {
            int notTake = 0 + dp[ind - 1][w];
            int take = INT_MIN;
            if (w >= wt[ind]) take = val[ind] + dp[ind - 1][w - wt[ind]];

            dp[ind][w] = max(take, notTake);
        }
    }

    return dp[n - 1][W];
}

int getValidPositiveInput(const string &message) {
    int input;
    while (true) {
        cout << message;
        cin >> input;
        if (cin.good() && input > 0) {
            break;
        } else {
            cout << "Invalid input. Please enter a valid positive integer.\n";
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
        }
    }
    return input;
}

void displayResults(int n, int W, const vector<int> &wt, const vector<int> &val) {
    cout << "Number of items: " << n << "\n";
    cout << "Knapsack capacity: " << W << "\n";
    cout << "Weights of items: ";
    for (int i = 0; i < n; ++i) {
        cout << wt[i] << " ";
    }
    cout << "\nValues of items: ";
    for (int i = 0; i < n; ++i) {
        cout << val[i] << " ";
    }
    cout << "\n";
}

void solveKnapsack(int n, int W, const vector<int> &wt, const vector<int> &val) {
    cout << "Solving 0-1 Knapsack problem...\n";

    cout << "Maximum Profit (Tabulated Knapsack): " << tabulatedKnapsack(n, W, wt, val) << "\n";
}

int main() {
    int choice;
    do {
        cout << "Menu:\n";
        cout << "1. Solve 0-1 Knapsack Problem\n";
        cout << "0. Exit\n";
        cout << "Enter your choice: ";

        choice = getValidPositiveInput("");

        switch (choice) {
            case 1:
                {
                    int n = getValidPositiveInput("Enter number of items: ");
                    int W = getValidPositiveInput("Enter Knapsack capacity: ");

                    vector<int> wt(n), val(n);
                    for (int i = 0; i < n; i++) {
                        wt[i] = getValidPositiveInput("Enter weight for item " + to_string(i + 1) + ": ");
                    }
                    for (int i = 0; i < n; i++) {
                        val[i] = getValidPositiveInput("Enter value for item " + to_string(i + 1) + ": ");
                    }

                    displayResults(n, W, wt, val);
                    solveKnapsack(n, W, wt, val);
                    break;
                }
            case 0:
                cout << "Exiting the program.\n";
                break;
            default:
                cout << "Invalid choice. Please try again.\n";
        }
    } while (choice != 0);

    return 0;
}
/*
    Input ->
    N = 5 
    W = 100
    Weights = {20, 24, 36, 40, 42}
    Values = {12, 35, 41, 25, 32}

    Ouptut -> 101
*/

/*
    Complexity Analysis ->
    Memoization -> T.C -> O(N*W)
                   S.C -> O(N*W) + O(N)
    Tabulation  -> T.C -> O(N*W)
                   S.C -> O(N*W)

*/
