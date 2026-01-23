#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>
#include <utility> 
using namespace std;
const long double INF = 1e9;
long double dist(pair<long double, long double> a, pair<long double, long double> b){
    return sqrt((a.first-b.first)*(a.first-b.first) + (a.second - b.second)*(a.second-b.second));
}

int main(){
    int n; cin >> n;
    vector<pair<long double, long double>> points(n);
    for(int i = 0; i < n; ++i){
        cin >> points[i].first >> points[i].second;
    } 

    vector<vector<long double>> d(n, vector<long double>(n));
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            d[i][j] = dist(points[i], points[j]);
        }
    }

    int N = 1 << n;
    vector<vector<long double>> dp(N, vector<long double>(n, INF));

    dp[1][0] = 0;

    for(int mask = 1; mask < N; ++mask){
        for(int u = 0; u < n; ++u){
            if(!(mask & (1 << u))) continue;
            long double cur = dp[mask][u];
            if(cur >= INF) continue;

            for(int v = 0; v < n; ++v){
                if(mask & (1 << v)) continue;
                int next_mask = mask | (1 << v);
                dp[next_mask][v] = min(dp[next_mask][v], cur + d[u][v]);
            }
        }
    }
    
    long double ans = INF;
    int full = N - 1;
    for(int i = 1; i < n; ++i){
        ans = min(ans, dp[full][i] + d[i][0]);
    }

    cout << fixed << setprecision(6) << ans << endl;
    return 0;
}