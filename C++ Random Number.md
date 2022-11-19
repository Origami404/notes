
````cpp
#include <random>
#include <iostream>

int main() {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(1.0, 10.0);

    for (int i=0; i<16; ++i)
        std::cout << dist(mt) << "\n";
}
````
