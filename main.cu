// !nvcc  -arch=sm_75 alm.cu --extended-lambda -o a.out -DANHN=2 -DSIZEL=1024

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/random.h>
#include <thrust/functional.h>
#include <cmath>
#include <iostream>
#include <fstream>

#ifndef ANHN
#define ANHN 2
#endif

#ifndef SIZEL
#define SIZEL 1024
#endif

typedef double real;

const int L = SIZEL;
const real Delta = 1.0f;
const real n = ANHN;
const real c = 0.0f;   // set c=0 for closed form

// -----------------------------
// Gaussian disorder
// -----------------------------
struct gaussian_rng {
    unsigned int seed;
    __host__ __device__
    real operator()(unsigned int i) const {
        thrust::default_random_engine rng(seed);
        rng.discard(i);
        thrust::normal_distribution<real> dist(0.0f, sqrtf(Delta));
        return dist(rng);
    }
};

// -----------------------------
// Constitutive law
// -----------------------------
struct slope_functor {
    real C;

    __host__ __device__
    real operator()(real F) const {
        real sigma = F + C;

        if (c == 0.0f) {
            return copysignf(powf(fabsf(sigma), 1.0f/(2.0f*n-1.0f)), sigma);
        }

        // Newton iteration (few steps, monotonic)
        real s = sigma / (c + 1.0f);
        for (int k=0; k<8; k++) {
            real g  = c*s + copysignf(powf(fabsf(s),2*n-1), s) - sigma;
            real gp = c + (2*n-1)*powf(fabsf(s),2*n-2);
            s -= g/gp;
        }
        return s;
    }
};


int main(int argc, char **argv) {

    unsigned seed = 1;
    if (argc > 1) {
        seed = atoi(argv[1]);
    }

    // -----------------------------
    // Disorder
    // -----------------------------
    thrust::device_vector<real> f(L);
    thrust::transform(
        thrust::counting_iterator<unsigned int>(0),
        thrust::counting_iterator<unsigned int>(L),
        f.begin(),
        gaussian_rng{seed}
    );
    real mean_f = thrust::reduce(f.begin(), f.end(), 0.0f) / L;
    thrust::transform(f.begin(), f.end(), f.begin(),
                      [=] __host__ __device__ (real x) { return x - mean_f; });    

    // -----------------------------
    // Prefix sum: cumulative force
    // -----------------------------
    thrust::device_vector<real> F(L);
    thrust::exclusive_scan(f.begin(), f.end(), F.begin());

    // -----------------------------
    // Find C by scalar root-finding
    // -----------------------------
    real C_lo = -50.0f, C_hi = 50.0f, C = 0.0f;

    for (int it=0; it<40; it++) {
        C = 0.5f*(C_lo + C_hi);

        real S = thrust::transform_reduce(
            F.begin(), F.end(),
            slope_functor{C},
            0.0f,
            thrust::plus<real>()
        );

        if (S > 0) C_hi = C;
        else       C_lo = C;
    }

    std::cout << "C = " << C << std::endl;

    // -----------------------------
    // Final slopes
    // -----------------------------
    thrust::device_vector<real> s(L);
    thrust::transform(F.begin(), F.end(), s.begin(), slope_functor{C});

    // -----------------------------
    // Reconstruct u
    // -----------------------------
    thrust::device_vector<real> u(L);
    thrust::exclusive_scan(s.begin(), s.end(), u.begin());

    real mean_u = thrust::reduce(u.begin(), u.end(), 0.0f) / L;
    thrust::transform(u.begin(), u.end(), u.begin(),
                      [=] __host__ __device__ (real x) { return x - mean_u; });

    real mean_s = thrust::reduce(s.begin(), s.end(), 0.0f) / L;
    std::cout << "Check periodicity (sum s): " << mean_s << std::endl;

    // Print first 10 elements of u
    thrust::host_vector<real> h_u_head(10);
    thrust::copy_n(u.begin(), 10, h_u_head.begin());

    std::cout << "First 10 elements of u: [";
    for (int i = 0; i < 10; ++i) {
        std::cout << h_u_head[i] << (i == 9 ? "" : ", ");
    }
    std::cout << "]" << std::endl;

    std::ofstream fout("u.txt");
    for (int i = 0; i < L; ++i) {
        fout << u[i] << std::endl;
    }

    std::cout << "Done." << std::endl;

    std::cout << "n=" << n << ", c=" << c << std::endl;
    std::cout << "L=" << L << ", Delta=" << Delta << std::endl;
    std::cout << "seed=" << seed << std::endl;

    return 0;
}
