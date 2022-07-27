#include <iostream>

using namespace std;

class A {
    public:
    int x;
    explicit A(int y) : x(y) {};
    ~A() {cout << "~A" << endl; };
};

class B : public A {
    public:
    int y;
    B(int a, int b=1) : A(a) {
        y = b;
    };
    // ~B() {cout << "~B" << endl;};
};

void tmp(int a, int b = 1) {
    cout << a  << b << endl;
}

int main() {
    if (1) {
        B b(1);
    }
    tmp(1);
    return 0;
}