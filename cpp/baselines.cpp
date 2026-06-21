#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <string>
using namespace std;

struct Point {
    double x, y;
};

double dist(const Point& a, const Point& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx*dx + dy*dy);
}

double tour_length(const vector<Point>& pts, const vector<int>& tour) {
    double total = 0.0;
    int n = tour.size();
    for (int i = 0; i < n; i++) {
        total += dist(pts[tour[i]], pts[tour[(i+1)%n]]);
    }
    return total;
}

vector<int> nearest_neighbor(const vector<Point>& pts) {
    int n = pts.size();
    vector<bool> visited(n, false);
    vector<int> tour;
    tour.reserve(n);

    int current = 0;
    tour.push_back(current);
    visited[current] = true;

    for (int step = 1; step < n; step++) {
        double best_d = 1e18;
        int best_j = -1;
        for (int j = 0; j < n; j++) {
            if (!visited[j]) {
                double d = dist(pts[current], pts[j]);
                if (d < best_d) {
                    best_d = d;
                    best_j = j;
                }
            }
        }
        current = best_j;
        visited[current] = true;
        tour.push_back(current);
    }
    return tour;
}

// Best-improvement 2-Opt including the wrap-around edge (tour[n-1], tour[0]).
// Each pass reads tour values FRESH (no stale pointers after a reversal),
// picks the single most-improving move, applies it, and repeats. Every accepted
// move strictly shortens the tour (gain > 1e-12), so this is guaranteed to terminate.
void two_opt(vector<int>& tour, const vector<Point>& pts) {
    int n = tour.size();
    while (true) {
        double best_gain = 1e-12;
        int best_i = -1, best_j = -1;
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 2; j < n; j++) {
                // break edges (tour[i],tour[i+1]) and (tour[j],tour[(j+1)%n])
                double d0 = dist(pts[tour[i]],     pts[tour[i + 1]])
                          + dist(pts[tour[j]],     pts[tour[(j + 1) % n]]);
                double d1 = dist(pts[tour[i]],     pts[tour[j]])
                          + dist(pts[tour[i + 1]], pts[tour[(j + 1) % n]]);
                double gain = d0 - d1;
                if (gain > best_gain) {
                    best_gain = gain;
                    best_i = i;
                    best_j = j;
                }
            }
        }
        if (best_i < 0) break;
        reverse(tour.begin() + best_i + 1, tour.begin() + best_j + 1);
    }
}

int main(int argc, char* argv[]) {
    // --per-instance: in addition to the averages, print one
    //   "PI <i> <nn_len> <2opt_len>" line per instance, so callers can compute
    //   their own mean/std. Default behaviour (no flag) is unchanged, and the
    //   average lines are always printed, so existing callers are unaffected.
    bool per_instance = (argc > 1 && string(argv[1]) == "--per-instance");

    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);
    cout << fixed << setprecision(6);   // set once, so PI lines are formatted too

    int num_instances, n;
    cin >> num_instances >> n;

    double total_nn = 0.0;
    double total_2opt = 0.0;

    for (int inst = 0; inst < num_instances; inst++) {
        vector<Point> pts(n);
        for (int i = 0; i < n; i++) {
            cin >> pts[i].x >> pts[i].y;
        }

        vector<int> tour_nn = nearest_neighbor(pts);
        double len_nn = tour_length(pts, tour_nn);

        vector<int> tour_2opt = tour_nn;
        two_opt(tour_2opt, pts);
        double len_2opt = tour_length(pts, tour_2opt);

        total_nn += len_nn;
        total_2opt += len_2opt;

        if (per_instance) {
            cout << "PI " << inst << " " << len_nn << " " << len_2opt << "\n";
        }
    }

    cout << "NN's model average tour: " << total_nn / num_instances << "\n";
    cout << "2OPT's model average tour: " << total_2opt / num_instances << "\n";

    return 0;
}