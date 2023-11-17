#include<bits/stdc++.h>

using namespace std;

struct MinHeapNode {
    char data;
    int freq;
    MinHeapNode *left, *right;
    
    MinHeapNode(char data, int freq) {
        this->data = data;
        this->freq = freq;
        this->left = nullptr;
        this->right = nullptr;
    }
};

struct compare {
    bool operator()(MinHeapNode* ele1, MinHeapNode* ele2) {
        return ele1->freq > ele2->freq;
    }
};

class HuffmanCoding {
private:
    string s;
    priority_queue<MinHeapNode*, vector<MinHeapNode*>, compare> minHeap;
    map<char,int> str;
    map<char, string> codes;

    void huffmanCodeHelper(map<char,int> &str) {
        for(auto node : str) {
            minHeap.push(new MinHeapNode(node.first, node.second));
        }

        MinHeapNode *left, *right;
        while(minHeap.size() != 1) {
            left = minHeap.top();
            minHeap.pop();
            right = minHeap.top();
            minHeap.pop();
            MinHeapNode *tmp = new MinHeapNode('$', left->freq + right->freq);
            tmp->left = left;
            tmp->right = right;
            minHeap.push(tmp);
        }

        getCodes(minHeap.top(), "");
    }

    void getCodes(MinHeapNode *root, string code){
        if(root == nullptr) return;

        if(root->data != '$') codes[root->data] = code;

        getCodes(root->left, code + '0');
        getCodes(root->right, code + '1');
    }

    void printCodes(map<char, string> &codes) {
        for(auto x : codes) {
            cout << x.first << " -> " << x.second << "\n"; 
        }
    }

public: 
    HuffmanCoding(string s) {
        this->s = s;
        for(auto x : s) str[x]++;
    }

    HuffmanCoding(map<char, int> str){
        this->str = str;
    }

    void getHuffmanCodes() { 
        if(codes.empty()) huffmanCodeHelper(str);
        printCodes(codes);
    }

    void encryptString() {
        if(codes.empty()) huffmanCodeHelper(str);

        string es = "";
        for(auto ch : s) {
            es += codes[ch];
        }

        cout << "Encrypted string: " << es << "\n";
    }
};

int get_positive_integer_input(const string& message) {
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

char get_char_input(const string& message) {
    char input;
    cout << message;
    cin >> input;
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
    return input;
}

int main() {
    int choice;
    map<char, int> str;

    do {
        cout << "Menu:\n";
        cout << "1. Enter Character Frequencies\n";
        cout << "2. Get Huffman Tree and Codes\n";
        cout << "3. Encrypt String\n";
        cout << "0. Exit\n";
        cout << "Enter your choice: ";

        choice = get_positive_integer_input("");

        switch (choice) {
            case 1:
                {
                    int n = get_positive_integer_input("Enter number of characters: ");
                    str.clear();
                    for (int i = 0; i < n; i++) {
                        char ch = get_char_input("Enter character: ");
                        int freq = get_positive_integer_input("Enter frequency: ");
                        str[ch] = freq;
                    }
                    break;
                }
            case 2:
                if (str.empty()) {
                    cout << "Please enter character frequencies first.\n";
                } else {
                    HuffmanCoding h(str);
                    h.getHuffmanCodes();
                }
                break;
            case 3:
                if (str.empty()) {
                    cout << "Please enter character frequencies first.\n";
                } else {
                    HuffmanCoding h(str);
                    h.getHuffmanCodes();
                    h.encryptString();
                }
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
