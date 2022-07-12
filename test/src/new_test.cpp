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
    B(int a) : A(a) {};
    // ~B() {cout << "~B" << endl;};
};

int main() {
    if (1) {
        B b(1);
    }

    return 0;
}