#include<bits/stdc++.h>

using namespace std;

void printNQueensSolutions(int n, vector<vector<string>> &res) {
    for (auto x : res) {
        for (int i = 0; i < n; i++) 
            cout << x[i] << "\n";
        cout << "\n";
    }
}

void nQueens(int c, int n, map<int, bool> &row, map<int, bool> &ld, map<int, bool> &ud, vector<string> &mat, vector<vector<string>> &res) {
    if (c == n) {
        res.push_back(mat);
        return;
    }

    for (int r = 0; r < n; r++) {
        if (!row[r] && !ld[r + c] && !ud[n - 1 + c - r]) {
            mat[r][c] = 'Q';
            row[r] = true;
            ld[r + c] = true;
            ud[n - 1 + c - r] = true;

            nQueens(c + 1, n, row, ld, ud, mat, res);

            row[r] = false;
            ld[r + c] = false;
            ud[n - 1 + c - r] = false;
            mat[r][c] = '.';
        }
    }
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

int main() {
    int choice;
    do {
        cout << "Menu:\n";
        cout << "1. Solve N-Queens Problem\n";
        cout << "2. Exit\n";
        cout << "Enter your choice: ";

        choice = getValidPositiveInput("");

        switch (choice) {
            case 1:
                {
                    int n = getValidPositiveInput("Enter size: ");

                    vector<string> mat(n, string(n, '.'));
                    map<int, bool> row, ld, ud;
                    vector<vector<string>> res;

                    nQueens(0, n, row, ld, ud, mat, res);

                    cout << "Solutions for N-Queens Problem:\n";
                    printNQueensSolutions(n, res);
                    break;
                }
            case 2:
                cout << "Exiting the program.\n";
                break;
            default:
                cout << "Invalid choice. Please try again.\n";
        }
    } while (choice != 0);

    return 0;
}
