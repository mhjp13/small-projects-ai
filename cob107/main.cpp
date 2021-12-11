#include <iostream>
#include <set>
#include <queue>
#include <stack>
#include <iterator>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>
using namespace std;

#define N 3

vector< vector<int> > getNeighbourCoords(int x, int y);
vector<int> getEmptySpaceCoords(vector< vector<char> > array);
vector < vector<char> > generateNewGrid(int oldX, int oldY, int newX, int newY, vector< vector<char> > array);
set<vector< vector<char> >> dfs_search(vector< vector<char> > startState, char* file_name);
void writeGrid(vector< vector<char> > array, char* file_name);
string gridToString(vector< vector<char> > array);
bool isSolvable(string str);
void searchAlgorithm(vector< vector<char> > s1, vector< vector<char> > s2);
vector< vector<char> > stringToGrid(char *str);

int main(int argc, char *argv[]) {
    if (argc != 3) {
        cout << "Invalid number of parameters\n";
        return 0;
    } else if (strlen(argv[1]) != 9 or strlen(argv[2]) != 9) {
        cout << "Invalid parameters\n";
        return 0;
    }

    set<vector< vector<char> >> exploredS1;
    set<vector< vector<char> >> exploredS2;
    vector< vector<char> > s1 = {{'a', 'b', 'c'}, {'d','e','f'}, {'g','h','0'}};
    vector< vector<char> > s2 = {{'0', 'e', 'b'}, {'a','h','c'}, {'d','g','f'}};
    vector< vector<char> > s3 = {{'1', '8', '2'}, {'0','4','3'}, {'7','6','5'}};
    vector< vector<char> > s4 = {{'8', '1', '2'}, {'0','4','3'}, {'7','6','5'}};
    vector< vector<char> > s5 = stringToGrid(argv[1]);
    vector< vector<char> > s6 = stringToGrid(argv[2]);

    searchAlgorithm(s5, s6);
    // cout << '0' << "\n";
    // vector<vector< vector<char> >> intersection;
    // cout << "Number of states in S2: " << exploredS2.size() << "\n";
    // set_intersection(exploredS1.begin(), exploredS1.end(), exploredS2.begin(), exploredS2.end(), back_inserter(intersection));
    // cout << "Number of common states: " << intersection.size() << "\n";
}

void searchAlgorithm(vector< vector<char> > s1, vector< vector<char> > s2) {
    set<vector< vector<char> >> exploredS1;
    set<vector< vector<char> >> exploredS2;

    if (isSolvable(gridToString(s1)) == isSolvable(gridToString(s2))) {
        exploredS1 = dfs_search(s1, "R(S1) - R(S2) - R(S1 & S2).txt");
        cout << "Number of states in S1: " << exploredS1.size() << "\n";
        cout << "Number of states in S2: " << exploredS1.size() << "\n";
        cout << "Number of states in S1 and S2: " << exploredS1.size() << "\n";
    } else {
        exploredS1 = dfs_search(s1, "R(S1).txt");
        exploredS2 = dfs_search(s2, "R(S2).txt");
        cout << "Number of states in S1: " << exploredS1.size() << "\n";
        cout << "Number of states in S2: " << exploredS2.size() << "\n";
        cout << "Number of states in S1 and S2: " << '0' << "\n";
    }
}

set<vector< vector<char> >> dfs_search(vector< vector<char> > startState, char* file_name) {
    set<vector< vector<char> >> explored;
    stack< vector< vector<char> >> frontier;
    frontier.push(startState);
    explored.insert(startState);

    while (!frontier.empty()) {
        vector< vector<char> > grid = frontier.top();
        frontier.pop();
        vector<int> emptySpaceCoords = getEmptySpaceCoords(grid);
        vector< vector<int> > neighbours = getNeighbourCoords(emptySpaceCoords[0], emptySpaceCoords[1]);

        writeGrid(grid, file_name);
        for (int i = 0; i < neighbours.size(); i++) {
            vector < vector<char> > newGrid = generateNewGrid(emptySpaceCoords[0], emptySpaceCoords[1], neighbours[i][0], neighbours[i][1], grid);
            if (explored.find(newGrid) == explored.end()) {
                explored.insert(newGrid);
                frontier.push(newGrid);
            }
        }
    }

    return explored;
}

vector< vector<int> > getNeighbourCoords(int x, int y) {
    vector< vector<int> > neighbourCoords;
    if (x > 0)
        neighbourCoords.push_back({x-1, y});
    if (y > 0)
        neighbourCoords.push_back({x, y-1});
    if (y < N - 1)
        neighbourCoords.push_back({x, y+1});
    if (x < N - 1)
        neighbourCoords.push_back({x+1, y});
    
    return neighbourCoords;
}

vector < vector<char> > generateNewGrid(int oldX, int oldY, int newX, int newY, vector< vector<char> > array) {
    vector < vector<char> > newGrid;
    copy(array.begin(), array.end(), back_inserter(newGrid));

    char temp = newGrid[newX][newY];
    newGrid[newX][newY] = '0';
    newGrid[oldX][oldY] = temp;
    
    return newGrid;
}

vector<int> getEmptySpaceCoords(vector< vector<char> > array) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            if (array[i][j] == '0') return {i,j};
    }
}

string gridToString(vector< vector<char> > array) {
    string newString;
    newString.resize(9);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            newString[j + N*i] = array[i][j];
    }

    return newString;
}

vector< vector<char> > stringToGrid(char *str) {
    vector< vector<char> > newGrid = {{'-', '-', '-'}, {'-', '-', '-'}, {'-', '-', '-'}};
    for (int i = 0; i < 9; i++)
        newGrid[i / 3][i%3] = str[i];
    return newGrid;
}

bool isSolvable(string str) {
    int inv_count = 0;
    for (int i = 0; i < N*N - 1; i++) {
        for (int j = i+1; j < N*N; j++)
            if (str[j] != '0' && str[i] != '0' &&  str[i] > str[j])
                inv_count++;
    }

    return inv_count % 2 == 0;
}

void writeGrid(vector< vector<char> > array, char* file_name) {
    string filename(file_name);
    ofstream MyFile;

    MyFile.open(filename, std::ios_base::app);
    if (array.size() == N)
        for (int i = 0; i < N; i++)
        {
            MyFile << array[i][0] << " " << array[i][1] << " " << array[i][2] << "\n";
        }
    MyFile << "\n";
    MyFile.close();
}