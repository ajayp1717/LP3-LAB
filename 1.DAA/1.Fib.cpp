#include <iostream>
#include <limits>
#include <cstdlib>

using namespace std;

// Recursive Fibonacci
int fib_recursive(int n) {
    if (n <= 1)
        return n;
    return fib_recursive(n - 1) + fib_recursive(n - 2);
}

// Iterative Fibonacci
void fib_iterative(int n) {
    cout << "Using Iterative: ";
    int f0 = 0, f1 = 1;
    cout << f0 << " ";
    for (int i = 0; i < n; i++) {
        cout << f1 << " ";
        int tmp = f0 + f1;
        f0 = f1;
        f1 = tmp;
    }
    cout << "\n";
}

// Function to get a valid positive integer input from the user
int get_valid_positive_input(const string& message) {
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
    while (true) {
        cout << "Menu:\n";
        cout << "1. Calculate Fibonacci (Iterative)\n";
        cout << "2. Calculate Fibonacci (Recursive)\n";
        cout << "0. Exit\n";

        int choice = get_valid_positive_input("Enter your choice: ");

        switch (choice) {
            case 1:
                {
                    int n = get_valid_positive_input("Enter the limit: ");
                    fib_iterative(n);
                    break;
                }
            case 2:
                {
                    int n = get_valid_positive_input("Enter the limit: ");
                    cout << "Using Recursive: ";
                    for (int i = 0; i < n; i++) {
                        cout << fib_recursive(i) << " ";
                    }
                    cout << "\n";
                    break;
                }
            case 0:
                cout << "Exiting the program.\n";
                return 0;
            default:
                cout << "Invalid choice. Please try again.\n";
        }
    }

    return 0;
}
