#include<iostream>
#include <vector>
#include <unordered_set>
#include <cmath>
using namespace std;

int main(){
    freopen("input.txt", "r", stdin);
    int r, c;
    cin>>r>>c;
    vector<vector<int>> inputMat(r, vector<int>(c));
    for(int i=0;i<=6;i++){
        for(int j=0;j<=3;j++){

            cin>> inputMat[i][j];
        }
    }
    for(int i=0;i<=6;i++){
        for(int j=0;j<=3;j++){
            cout<< inputMat[i][j]<<" ";
        }
        cout<<endl;
    }
    vector<int> hash(7);
    for(int i=0;i<7;i++){
        hash[i] = (1*i+3)%7;
        cout<<hash[i]<<" ";
    }
    cout<<endl;
    vector<int> hashone(7);
    for(int i=0;i<7;i++){
        hashone[i] = (1*i+5)%7;
        cout<<hashone[i]<<" ";
    }
    cout<<endl;
    vector<int> hashtwo(7);
    for(int i=0;i<7;i++){
        hashtwo[i] = (1*i+7)%7;
        cout<<hashtwo[i]<<" ";
    }
    cout<<endl;
    int i=




}