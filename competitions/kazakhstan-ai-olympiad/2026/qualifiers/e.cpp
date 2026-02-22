#include <bits/stdc++.h>
using namespace std;
#define ll long long 

const ll MAXN = 2e5+123;
vector <ll> a(MAXN), t1(MAXN*4+2), t2(MAXN*4+2);
ll n;

void build1(ll v, ll tl, ll tr) {
  if(tl == tr) {
    t1[v] = a[tl]*a[tl];
    return;
  }
  ll tm = (tl+tr)/2;
  build1(2*v, tl, tm);
  build1(2*v+1, tm+1, tr);
  
  t1[v] = t1[2*v] + t1[2*v+1];
}

void build2(ll v, ll tl, ll tr) {
  if(tl == tr) {
    t2[v] = a[tl];
    return;
  }
  ll tm = (tl+tr)/2;
  build2(2*v, tl, tm);
  build2(2*v+1, tm+1, tr);
  
  t2[v] = t2[2*v] + t2[2*v+1];
}

ll answer1(ll v, ll tl, ll tr, ll l, ll r) {
  if(tl > r || tr < l) {
    return 0ll;
  }
  if(tl >= l && tr <= r) {
    return t1[v];
  }
  ll tm = (tl+tr)/2;
  return answer1(2*v, tl, tm, l, r) + answer1(2*v+1, tm+1, tr, l, r);
}

ll answer2(ll v, ll tl, ll tr, ll l, ll r) {
  if(tl > r || tr < l) {
    return 0ll;
  }
  if(tl >= l && tr <= r) {
    return t2[v];
  }
  ll tm = (tl+tr)/2;
  return answer2(2*v, tl, tm, l, r) + answer2(2*v+1, tm+1, tr, l, r);
}


void solve() {
  ll q; cin >> n >> q;
  for(ll i=1; i<=n; i++) cin >> a[i];
  
  build1(1, 1, n);
  build2(1, 1, n);
  
  while(q--) {
    ll l, r; cin >> l >> r;
    ll sum_squared = answer1(1, 1, n, l, r);
    ll sum_numbers = answer2(1, 1, n, l, r);
    ll k = r-l+1;
    cout << k * sum_squared - sum_numbers * sum_numbers << "\n";
  }
}

int main() {
  
  int t = 1;
  while(t--) solve();
  
}
