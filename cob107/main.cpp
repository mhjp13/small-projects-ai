#include <iostream>
#include <set>
#include <queue>
#include <stack>
#include <iterator>
#include <vector>
#include <algorithm>
#include <fstream>
using namespace std;

#define N 3

void printGrid(vector< vector<char> > array);
vector< vector<int> > getNeighbourCoords(int x, int y);
void printCoords(vector< vector<int> > coords);
vector<int> getEmptySpaceCoords(vector< vector<char> > array);
vector < vector<char> > generateNewGrid(int oldX, int oldY, int newX, int newY, vector< vector<char> > array);
void dfs_search(vector< vector<char> > array);
void writeGrid(vector< vector<char> > array);

vector< vector<char> > s1 = {{'a', 'b', 'c'}, {'d','e','f'}, {'g','h','0'}};
vector< vector<char> > s2 = {{'0', 'e', 'b'}, {'a','h','c'}, {'d','g','f'}};
set<vector< vector<char> >> common_states;

int main(int argc, char *argv[]) {
    dfs_search(s1);
}

void dfs_search(vector< vector<char> > startState) {
    set<vector< vector<char> >> explored;
    stack< vector< vector<char> >> frontier;
    frontier.push(startState);
    explored.insert(startState);

    while (!frontier.empty()) {
        vector< vector<char> > grid = frontier.top();
        frontier.pop();
        vector<int> emptySpaceCoords = getEmptySpaceCoords(grid);
        vector< vector<int> > neighbours = getNeighbourCoords(emptySpaceCoords[0], emptySpaceCoords[1]);

        writeGrid(grid);
        for (int i = 0; i < neighbours.size(); i++) {
            vector < vector<char> > newGrid = generateNewGrid(emptySpaceCoords[0], emptySpaceCoords[1], neighbours[i][0], neighbours[i][1], grid);
            if (explored.find(newGrid) == explored.end()) {
                explored.insert(newGrid);
                frontier.push(newGrid);
            }
            common_states.insert(newGrid);
        }
    }

    printf("\nStates reached: %lu\n", explored.size());
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

void printGrid(vector< vector<char> > array) {
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            printf("%c ", array[i][j]);
        printf("\n");
    }
}

void writeGrid(vector< vector<char> > array) {
    string filename("R(S1).txt");
    ofstream MyFile;

    MyFile.open(filename, std::ios_base::app);
    for (int i = 0; i < N; i++)
    {
        MyFile << array[i][0] << " " << array[i][1] << " " << array[i][2] << "\n";
    }
    MyFile << "  ↓\n";
    MyFile.close();
}

void printCoords(vector< vector<int> > coords) {
    for (int i = 0; i < coords.size(); i++) {
        printf("(%d, %d)\n", coords[i][0], coords[i][1]);
    }
}