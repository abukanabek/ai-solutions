#include <bits/stdc++.h>
using namespace std;
#define ll long long 
#define all(a) a.begin(), a.end()

struct Token {
  ll q, k, v, ind;
};

void solve() {
  ll n, T; cin >> n >> T;
  vector <Token> a(n);
  for(ll i=0; i<n; i++) {
    cin >> a[i].q >> a[i].k >> a[i].v;
    a[i].ind = i;
  }
  
  sort(all(a), [](const Token &x, const Token &y) {
    return x.k < y.k;
  });
  
  vector <ll> pref(n, 0);
  pref[0] = a[0].k * a[0].v;
  for(ll i=1; i<n; i++) {
    pref[i] = pref[i-1] + a[i].k * a[i].v;
  }
  
  vector <ll> suf(n, 0);
  suf[n-1] = a[n-1].v;
  for(ll i=n-2; i>=0; i--) {
    suf[i] = suf[i+1] + a[i].v;
  }
  
  vector <ll> ans(n, 0);
  
  for(ll i=0; i<n; i++) {
    ll cur = 0;
    
    ll l=0, r=n-1;
    while(l <= r) {
      ll m = (l+r)/2;
      if(a[i].q * a[m].k >= T) r = m-1;
      else l = m+1;
    }
    ll edge=-1;
    if(l-1 >= 0 && a[i].q * a[l-1].k >= T) edge = l-1;
    else if(l >= 0 && a[i].q * a[l].k >= T) edge = l;
    else if(l+1 < n && a[i].q * a[l+1].k >= T) edge = l+1;
    // cout << l << " ";
    // cout << a[i].q << " " << a[i].k << " " << a[i].v << " " << a[i].ind << " " << edge << "\n";
    
    if(edge == -1) {
      cur += pref[n-1] * a[i].q;
    }
    else {
      if(edge-1 >= 0) {
        cur += pref[edge-1] * a[i].q;
      }
      cur += suf[edge] * T;
    }
    
    ans[a[i].ind] = cur;
  }
  
  for(ll i=0; i<n; i++) cout << ans[i] << " ";
}

int main() {
  
  int t = 1;
  while(t--) solve();
  
}
