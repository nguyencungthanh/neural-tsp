#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>
using namespace std; 

int main(int argc, char* argv[]) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    // Debug 
    if (argc < 4) {
        cout << "Usage: ./generate_data <num_instances> <num_points> <output_file>\n";
        return 1;
    }

    int num_instances = stoi(argv[1]);
    int n = stoi(argv[2]);
    string output_file = argv[3];

    ofstream out(output_file);
    if (!out) {
        cerr << "Error opening file for writing.\n";
        return 1;
    }

    // Random generator
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    out << num_instances << " " << n << "\n";
    out << fixed << setprecision(6);

    for (int i = 0; i < num_instances; ++i) {
        for (int j = 0; j < n; ++j) {
            double x = dis(gen);
            double y = dis(gen);
            out << x << " " << y << "\n";
        }
    }

    out.close();
    cout << "Generated " << num_instances << " TSP instances with " << n << " points each.\n";
    cout << "Saved to " << output_file << "\n";

    return 0;
}
