#include <iostream>
#include <vector>
#include <limits>
#include <ctime>
#include <cstdlib>

using namespace std;

// Function to partition the array
template<typename T>
int partition(vector<T>& arr, int low, int high) {
    T pivot = arr[low];
    int i = low + 1;
    int j = high;

    while (true) {
        while (i <= j && arr[i] <= pivot) {
            i++;
        }
        while (i <= j && arr[j] > pivot) {
            j--;
        }
        if (i <= j) {
            swap(arr[i], arr[j]);
        } else {
            break;
        }
    }

    swap(arr[low], arr[j]);
    return j;
}

// Deterministic QuickSort
template<typename T>
void deterministic_quick_sort(vector<T>& arr, int low, int high) {
    if (low < high) {
        int pivot_index = partition(arr, low, high);
        deterministic_quick_sort(arr, low, pivot_index - 1);
        deterministic_quick_sort(arr, pivot_index + 1, high);
    }
}

// Randomized QuickSort
template<typename T>
void randomized_quick_sort(vector<T>& arr, int low, int high) {
    if (low < high) {
        // Randomly choose a pivot and swap it with the first element
        int random_index = low + rand() % (high - low + 1);
        swap(arr[low], arr[random_index]);

        int pivot_index = partition(arr, low, high);
        randomized_quick_sort(arr, low, pivot_index - 1);
        randomized_quick_sort(arr, pivot_index + 1, high);
    }
}

// Function to display the array
template<typename T>
void display_array(const vector<T>& arr) {
    for (const T& num : arr) {
        cout << num << " ";
    }
    cout << endl;
}

// Function to get a valid input of type T from the user
template<typename T>
T get_valid_input() {
    T input;
    while (!(cin >> input) || cin.fail()) {
        cin.clear(); // clear input buffer to restore cin to a usable state
        cin.ignore(numeric_limits<streamsize>::max(), '\n'); // ignore last input
        cout << "Invalid input. Please enter a valid value: ";
    }
    return input;
}

int main() {
    srand(static_cast<unsigned int>(time(0))); // Seed for random number generation

    vector<int> array_to_sort;

    int choice;
    do {
        cout << "Menu:\n";
        cout << "1. Enter Array\n";
        cout << "2. Deterministic QuickSort\n";
        cout << "3. Randomized QuickSort\n";
        cout << "4. Display Array\n";
        cout << "0. Exit\n";
        cout << "Enter your choice: ";
        choice = get_valid_input<int>();

        switch (choice) {
            case 1:
                int size;
                do {
                    cout << "Enter the size of the array (non-negative): ";
                    size = get_valid_input<int>();
                    if (size < 0) {
                        cout << "Invalid size. Please enter a non-negative value.\n";
                    }
                } while (size < 0);

                array_to_sort.resize(size);

                cout << "Enter the elements of the array:\n";
                for (int i = 0; i < size; ++i) {
                    cout << "Element " << i + 1 << ": ";
                    array_to_sort[i] = get_valid_input<int>();
                }
                break;

            case 2:
                deterministic_quick_sort(array_to_sort, 0, array_to_sort.size() - 1);
                cout << "Deterministic QuickSort Result: ";
                display_array(array_to_sort);
                break;

            case 3:
                randomized_quick_sort(array_to_sort, 0, array_to_sort.size() - 1);
                cout << "Randomized QuickSort Result: ";
                display_array(array_to_sort);
                break;

            case 4:
                cout << "Current Array: ";
                display_array(array_to_sort);
                break;

            case 0:
                cout << "Exiting the program.\n";
                break;

            default:
                cout << "Invalid choice. Please try again.\n";
        }
    } while (choice != 0);

    return 0;
}
