#include <iostream>
#include <vector>
#include <cmath>
#include <utility> 
using namespace std;

struct Point { double x, y; };

double dist(const Point& a, const Point& b){
    double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx*dx + dy*dy);
}

pair<double, vector<int>> held_karp(const vector<Point>& pts) {
    int n = pts.size();
    int N = 1 << n;

    vector<vector<double>> dp(N, vector<double>(n, 1e18));
    vector<vector<int>> parent(N, vector<int>(n, -1));

    dp[1][0] = 0;

    for (int mask = 1; mask < N; mask++) {
        for (int u = 0; u < n; u++) {
            if (!(mask & (1<<u))) continue;
            double cur = dp[mask][u];
            if (cur >= 1e18) continue;

            for (int v = 0; v < n; v++) {
                if (mask & (1<<v)) continue;
                int next_mask = mask | (1<<v);
                double nd = cur + dist(pts[u], pts[v]);
                if (nd < dp[next_mask][v]) {
                    dp[next_mask][v] = nd;
                    parent[next_mask][v] = u;
                }
            }
        }
    }

    int full = N - 1;
    double best = 1e18;
    int last = -1;

    for (int u = 1; u < n; u++) {
        double cand = dp[full][u] + dist(pts[u], pts[0]);
        if (cand < best) {
            best = cand;
            last = u;
        }
    }

    vector<int> tour(n);
    int mask = full;
    int cur = last;

    for (int i = n-1; i >= 1; i--) {
        tour[i] = cur;
        int p = parent[mask][cur];
        mask ^= (1<<cur);
        cur = p;
    }
    tour[0] = 0;

    return {best, tour};
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);
    
    int num_instances, n;
    cin >> num_instances >> n;

    cout << num_instances << " " << n << "\n";

    for (int inst = 0; inst < num_instances; inst++) {
        vector<Point> pts(n);
        for (int i = 0; i < n; i++)
            cin >> pts[i].x >> pts[i].y;

        auto result = held_karp(pts);
        auto& tour = result.second;

        // Store the coordinates
        for (auto &p : pts)
            cout << p.x << " " << p.y << "\n";

        // Store the optimized tour
        for (int i = 0; i < n; i++)
            cout << tour[i] << (i+1<n ? ' ' : '\n');
    }
}