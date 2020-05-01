#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>
#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif

using ptr = std::unique_ptr<double[]>;

// See README.md for instructions and information.

void plot(int const nel, double const *ndx, double const *el)
{
    for (int iel = 0; iel < nel; iel++)
    {
        int const indl = iel;
        int const indr = iel + 1;
        double const cx = 0.5 * (ndx[indl] + ndx[indr]);
        std::printf("%12.6e,%12.6e\n", cx, el[iel]);
    }
}

// struct wrap
// {
//     ptr _ndx;    // node positions
//     ptr _ndx05;  // half-step node positions
//     ptr _ndm;    // Lagrangian nodal masses
//     ptr _ndu;    // nodal velocities
//     ptr _ndubar; // nodal timestep-average velocities
//     ptr _elrho;  // cell densities
//     ptr _elp;    // cell pressures
//     ptr _elq;    // cell artificial viscosities
//     ptr _elein;  // cell specific internal energies
//     ptr _elv;    // cell volumes (lengths)
//     ptr _elm; 
//     const double _dt;
// };


int main(int const argc, char const *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <num_cells>" << std::endl;
        return 1;
    }

    double constexpr GAMMA = 1.4; // Material constant.

    // 1D mesh with user-specified number of cells, more cells is more accurate.
    const int nel = std::stoi(argv[1]);
    const int nnd = nel + 1;

    ptr ndx(new double[nnd]);    // node positions
    ptr ndx05(new double[nnd]);  // half-step node positions
    ptr ndm(new double[nnd]);    // Lagrangian nodal masses
    ptr ndu(new double[nnd]);    // nodal velocities
    ptr ndubar(new double[nnd]); // nodal timestep-average velocities
    ptr elrho(new double[nel]);  // cell densities
    ptr elp(new double[nel]);    // cell pressures
    ptr elq(new double[nel]);    // cell artificial viscosities
    ptr elein(new double[nel]);  // cell specific internal energies
    ptr elv(new double[nel]);    // cell volumes (lengths)
    ptr elm(new double[nel]);    // Lagrangian cell masses

    // double ndx[nnd];
    // double ndx05[nnd];  // half-step node positions
    // double ndm[nnd];    // Lagrangian nodal masses
    // double ndu[nnd];    // nodal velocities
    // double ndubar[nnd]; // nodal timestep-average velocities
    // double elrho[nel];  // cell densities
    // double elp[nel];    // cell pressures
    // double elq[nel];    // cell artificial viscosities
    // double elein[nel];  // cell specific internal energies
    // double elv[nel];    // cell volumes (lengths)
    // double elm[nel];

    // XXX --- MPI version needs a 1 cell "ghost layer" for elm, elp and elq ---

    // Initialise node positions (equally spaced, x \in [0,1]).
#ifdef _OPENMP
#pragma omp parallel firstprivate(nnd, nel)
    {
#pragma omp for
        for (int ind = 0; ind < nnd; ind++)
            ndx[ind] = (1.0 / nel) * ind;
    }
#else
    for (int ind = 0; ind < nnd; ind++)
        ndx[ind] = (1.0 / nel) * ind;
#endif

    // Initial conditions for Sod's shock tube (left and right states).
    for (int iel = 0; iel < nel; iel++)
    {
        int const indl = iel;
        int const indr = iel + 1;
        bool const left = (0.5 * (ndx[indl] + ndx[indr]) < 0.5);
        elrho[iel] = left ? 1.0 : 0.125;
        elp[iel] = left ? 1.0 : 0.1;
        elein[iel] = elp[iel] / (elrho[iel] * (GAMMA - 1.0));
        elv[iel] = ndx[indr] - ndx[indl];
        elm[iel] = elrho[iel] * elv[iel];
    }

    // XXX --- MPI comms needed here to handle setting ndm correctly below. ---

    for (int ind = 0; ind < nnd; ind++)
    {
        int const iell = std::max(ind - 1, 0);
        int const ielr = std::min(ind, nel - 1);

        ndu[ind] = 0.0;
        ndm[ind] = 0.5 * (elm[iell] + elm[ielr]);
    }
    std::clock_t c_start = std::clock();
    auto t_start = std::chrono::high_resolution_clock::now();
    // Main timestepping loop, t \in [0,0.25]. Use an explicit finite element
    // discretisation to solve the compressible Euler equations.
    int istep = 1;
    double t = 0.0, dt = 1.0e-6;
    while (t < 0.25)
    {
        // Calculate artificial viscosity (Q) and minimum CFL condition.
        double min_cfl = std::numeric_limits<double>::max();
#ifdef _OPENMP
#pragma omp parallel shared(ndu, elein, elrho, elv, elq, min_cfl), firstprivate(GAMMA, nel)
        {
#pragma omp for reduction(min:min_cfl) schedule(static)
            for (int iel = 0; iel < nel; iel++)
            {
                int const indl = iel;
                int const indr = iel + 1;

                // Scalar Q (unlimited).
                double const du = std::min(ndu[indr] - ndu[indl], 0.0);
                double const c2 = GAMMA * (GAMMA - 1.0) * elein[iel];
                elq[iel] = 0.75 * elrho[iel] * du * du +
                           0.5 * elrho[iel] * std::sqrt(c2) * std::fabs(du);

                // CFL condition (with Q correction to sound speed).
                double const c2_corr = c2 + 2.0 * elq[iel];
                double const cfl = elv[iel] / std::max(1.0e-40, std::sqrt(c2_corr));
                min_cfl = std::min(min_cfl, cfl);

                // XXX --- Need to reduce min_cfl over OpenMP threads ---
            }
        }
#else
        for (int iel = 0; iel < nel; iel++)
        {
            int const indl = iel;
            int const indr = iel + 1;

            // Scalar Q (unlimited).
            double const du = std::min(ndu[indr] - ndu[indl], 0.0);
            double const c2 = GAMMA * (GAMMA - 1.0) * elein[iel];
            elq[iel] = 0.75 * elrho[iel] * du * du +
                       0.5 * elrho[iel] * std::sqrt(c2) * std::fabs(du);

            // CFL condition (with Q correction to sound speed).
            double const c2_corr = c2 + 2.0 * elq[iel];
            double const cfl = elv[iel] / std::max(1.0e-40, std::sqrt(c2_corr));
            min_cfl = std::min(min_cfl, cfl);

            // XXX --- Need to reduce min_cfl over OpenMP threads ---
        }
#endif

        // Get timestep.

        dt = std::min({0.25 - dt, dt * 1.02, 0.5 * min_cfl});

        // XXX --- MPI comms needed here to get global min. timestep ---

        // Predict half-step geometry and calculate pressure.

        int rank = 0;
        int size = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        for (int ind = 0; ind < nnd; ind++)
        {
            ndx05[ind] = ndx[ind] + 0.5 * dt * ndu[ind];
        }

        for (int iel = 0; iel < nel; iel++)
        {
            int const indl = iel;
            int const indr = iel + 1;
            elv[iel] = ndx05[indr] - ndx05[indl];
            double const rho = elm[iel] / elv[iel];
            double const idivu = ndu[indr] - ndu[indl];
            double const ein =
                elein[iel] - 0.5 * dt * (elp[iel] + elq[iel]) * idivu / elm[iel];
            elp[iel] = (GAMMA - 1.0) * ein * rho;
        }

        // XXX --- MPI comms needed here to exchange elp and elq ghosts ---

        // Corrector step to obtain full-step geometry and thermodynamic states.
        for (int ind = 0; ind < nnd; ind++)
        {
            int const iell = ind - 1;
            int const ielr = ind;

            // Force on node due to neighbour element pressures.
            double f = 0.0;
            if (iell >= 0)
                f += elp[iell] + elq[iell];
            if (ielr < nel)
                f -= elp[ielr] + elq[ielr];

            // Calculate a=F/m and apply zero-acceleration boundary conditions.
            double a = f / std::max(1.0e-40, ndm[ind]);
            if (ind == 0 || ind == nnd - 1)
                a = 0.0;

            // XXX --- Need to correctly handle boundaries with MPI decomp. ---

            // Update velocity and position.
            double const uprev = ndu[ind];
            ndu[ind] += dt * a;
            ndubar[ind] = 0.5 * (uprev + ndu[ind]);
            ndx[ind] += dt * ndubar[ind];
        }

        for (int iel = 0; iel < nel; iel++)
        {
            int const indl = iel;
            int const indr = iel + 1;
            elv[iel] = ndx[indr] - ndx[indl];

            double const idivu = ndubar[indr] - ndubar[indl];
            elein[iel] -= dt * (elp[iel] + elq[iel]) * idivu / elm[iel];
            elrho[iel] = elm[iel] / elv[iel];
            elp[iel] = (GAMMA - 1.0) * elein[iel] * elrho[iel];
        }

        // Step I/O.
        if (istep % 100 == 0)
        {
            char buf[256];
            std::snprintf(buf, 256, "step=%8d\tt=%12.6e\tdt=%12.6e", istep, t, dt);
            std::cerr << std::string(buf) << std::endl;
        }

        t += dt;
        istep++;
    }
    std::clock_t c_end = std::clock();
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "Original code time usage." << std::endl;
    std::cout << std::fixed << "CPU time used: "
              << 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC << " ms\n"
              << "Wall clock time passed: "
              << std::chrono::duration<double, std::milli>(t_end - t_start).count()
              << " ms\n";
    // XXX --- Uncomment this line to write density data to stdout. ---
    // XXX --- MPI comms needed here to gather data to root process. ---
    // plot(nel, ndx.get(), elrho.get());
    return 0;
}
