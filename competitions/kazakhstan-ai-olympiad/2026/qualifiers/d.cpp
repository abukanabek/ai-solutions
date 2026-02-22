#include <bits/stdc++.h>
using namespace std;
#define ll long long 
#define dd double
#define ld long double

dd e = 2.7182818;

dd sigmoid(dd x) {
  return ((dd) (1 / (1 + pow(e, -x))));
}
void solve() {
  int p, n; cin >> p >> n;
  dd x = ((dd) (p / 1e8));
  for(int i=0; i<n; i++) {
    x = sigmoid(x);
    if(abs(x - sigmoid(x)) < 1e-7) {
      break;
    }
  }
  cout << fixed << setprecision(9) << x << "\n";
}

int main() {
  
  int t = 1; cin >> t;
  while(t--) solve();
  
}
