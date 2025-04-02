#include <iostream>
#include <omp.h>
#include <vector>
using namespace std;

bool isPrime(int num) {
    if (num <= 1) return false;
    if (num <= 3) return true;
    if (num % 2 == 0 || num % 3 == 0) return false;
    for (int i = 5; i * i <= num; i += 6) {
        if (num % i == 0 || num % (i + 2) == 0) return false;
    }
    return true;
}

int main() {
    int start, end;
    cout << "Enter the start of the range: ";
    cin >> start;
    cout << "Enter the end of the range: ";
    cin >> end;

    vector<int> primes;

    #pragma omp parallel for
    for (int i = start; i <= end; ++i) {
        if (isPrime(i)) {
            #pragma omp critical
            primes.push_back(i);
        }
    }

    cout << "Prime numbers between " << start << " and " << end << " are: ";
    for (int prime : primes) {
        cout << prime << " ";
    }
    cout << endl;

    return 0;
}