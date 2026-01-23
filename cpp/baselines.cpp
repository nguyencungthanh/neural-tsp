#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
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

void two_opt(vector<int>& tour, const vector<Point>& pts) {
    int n = tour.size();
    bool improved = true;

    while (improved) {
        improved = false;
        for (int i = 1; i < n-2; i++) {
            for (int k = i+1; k < n-1; k++) {
                int a = tour[i-1];
                int b = tour[i];
                int c = tour[k];
                int d = tour[(k+1)%n];

                double old_dist = dist(pts[a], pts[b]) + dist(pts[c], pts[d]);
                double new_dist = dist(pts[a], pts[c]) + dist(pts[b], pts[d]);

                if (new_dist < old_dist) {
                    reverse(tour.begin()+i, tour.begin()+k+1);
                    improved = true;
                }
            }
        }
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);   
    cout.tie(nullptr);
    
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
    }

    cout << fixed << setprecision(6);
    cout << "NN " << total_nn / num_instances << "\n";
    cout << "2OPT " << total_2opt / num_instances << "\n";

    return 0;
}

