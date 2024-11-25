#include <NTL/ZZ_pX.h>
#include <NTL/mat_ZZ_p.h>
#include <NTL/mat_ZZ.h>
#include <random>
#include <vector>
#include <chrono>
#include <cassert>
using namespace std;
using namespace NTL;

extern "C" {
    void get_sk(vector<long>& vec, vector<long>& vec_inv, long Q, int N)
    {
        vec = vector<long>(N, 0); vec_inv = vector<long>(N, 0);
        ZZ_p::init(ZZ(Q));
        ZZ_pX poly, inv_poly, modulus;
        poly = inv_poly = modulus = 0;
        SetCoeff(modulus, N, 1);
        SetCoeff(modulus, 0, 1);
        static uniform_int_distribution<int> ternary_sampler(-1, 1);
        static default_random_engine rand_engine(std::chrono::system_clock::now().time_since_epoch().count());
        while (true)
        {
            for (size_t i = 0; i < N; i++)
                SetCoeff(poly, i, ternary_sampler(rand_engine));
            try
            {
                InvMod(inv_poly, poly, modulus);
                break;
            }
            catch(...)
            {
                cout << "Polynomial " << poly << " isn't a unit" << endl;
                continue;
            }
        }
        long tmp_coef;
        for (int i = 0; i <= deg(poly); i++)
        {
            tmp_coef = conv<long>(poly[i]);
            if( tmp_coef < 0 )
                tmp_coef += Q;
            vec[i] = tmp_coef;
        }
        for (int i = 0; i <= deg(inv_poly); i++)
        {
            tmp_coef = conv<long>(inv_poly[i]);
            if( tmp_coef < 0 )
                tmp_coef += Q;
            vec_inv[i] = tmp_coef;
        }
    }
}