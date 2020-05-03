#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <mpi.h>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

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

// struct Wrap
// {
//     double *_ndx;    // node positions
//     double *_ndx05;  // half-step node positions
//     double *_ndm;    // Lagrangian nodal masses
//     double *_ndu;    // nodal velocities
//     double *_ndubar; // nodal timestep-average velocities
//     double *_elrho;  // cell densities
//     double *_elp;    // cell pressures
//     double *_elq;    // cell artificial viscosities
//     double *_elein;  // cell specific internal energies
//     double *_elv;    // cell volumes (lengths)
//     double *_elm;
//     double _dt;
//     int _nel;
//     int _nnd;
//     int indStart; //index start
//     int indEnd;   //index end
// };

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <num_cells>" << std::endl;
        return 1;
    }

    double constexpr GAMMA = 1.4; // Material constant.

    // 1D mesh with user-specified number of cells, more cells is more accurate.
    const int nelTotal = std::stoi(argv[1]);

    int nelIndexes[2];
    int nelIndexList[size * 2];
    int error;
    if (rank == 0)
    {
        auto blockSize = int(std::ceil(double(nelTotal) / size));

        for (auto i = 0; i < size; i++)
        {
            nelIndexList[i * 2 + 1] = blockSize * (i + 1) - 1;
            nelIndexList[i * 2] = blockSize * i;
        }
        nelIndexList[size * 2 - 1] = nelTotal - 1;

        for (auto i = 0; i < size * 2; i++)
        {
            std::cout << nelIndexList[i] << std::endl;
        }
    }
    error = MPI_Scatter((const void *)nelIndexList, 2, MPI_INT,
                        &nelIndexes, 2, MPI_INT, 0,
                        MPI_COMM_WORLD);

    const int nel = nelIndexes[1] - nelIndexes[0] + 1;
    const int nnd = nel + 1;

    using ptr = std::unique_ptr<double[]>;

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
    // Wrap origin;
    // origin._ndx = ndx.release();
    // origin._ndx05 = ndx05.release();
    // origin._ndm = ndm.release();
    // origin._ndu = ndu.release();
    // origin._ndubar = ndubar.release();
    // origin._elrho = elrho.release();
    // origin._elp = elp.release();
    // origin._elq = elq.release();
    // origin._elein = elein.release();
    // origin._elv = elv.release();
    // origin._elm = elm.release();
    // origin._nel = nel;
    // origin._nnd = nnd;

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

    double elmSend;
    double elpSend;
    double elqSend;

    double elmGhost;
    double elpGhost;
    double elqGhost;

    // XXX --- MPI version needs a 1 cell "ghost layer" for elm, elp and elq ---

    // std::clock_t c_start = std::clock();
    // auto t_start = std::chrono::high_resolution_clock::now();

    // Initialise node positions (equally spaced, x \in [0,1]).

    for (int ind = 0; ind < nnd; ind++)
        ndx[ind] = (1.0 / nelTotal) * ind + nelIndexes[0];

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

    int tag_send = 0;
    int tag_recv = tag_send;

    // XXX --- MPI comms needed here to handle setting ndm correctly below. ---

    for (int ind = 0; ind < nnd; ind++)
    {
        int const iell = std::max(ind - 1, 0);
        int const ielr = std::min(ind, nel - 1);

        ndu[ind] = 0.0;
        if (ind == 0) //rank != 0 &&
        {
            elmSend = elm[nel - 1];

            error = MPI_Barrier(MPI_COMM_WORLD);

            for (auto i = 1; i < size; i++)
            {
                if (rank == (i - 1))
                    error = MPI_Send(&elmSend, 1, MPI_DOUBLE, i, tag_send, MPI_COMM_WORLD);
                else if (rank == i)
                    error = MPI_Recv(&elmGhost, 1, MPI_DOUBLE, i - 1, tag_recv, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            ndm[ind] = 0.5 * (elmGhost + elm[ielr]);
        }
        else if (ind == nel) // && rank != size - 1
        {
            elmSend = elm[0];
            error = MPI_Barrier(MPI_COMM_WORLD);
            for (auto i = 0; i < size - 1; i++)
            {
                if (rank == (i + 1))
                    error = MPI_Send(&elmSend, 1, MPI_DOUBLE, i, tag_send, MPI_COMM_WORLD);
                else if (rank == i)
                    error = MPI_Recv(&elmGhost, 1, MPI_DOUBLE, i + 1, tag_recv, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            ndm[ind] = 0.5 * (elm[iell] + elmGhost);
        }
        else
        {
            ndm[ind] = 0.5 * (elm[iell] + elm[ielr]);
        }
    }

    // Main timestepping loop, t \in [0,0.25]. Use an explicit finite element
    // discretisation to solve the compressible Euler equations.
    int istep = 1;
    double t = 0.0, dt = 1.0e-6;
    while (t < 0.25)
    {
        // Calculate artificial viscosity (Q) and minimum CFL condition.
        double min_cfl = std::numeric_limits<double>::max();
#ifdef _OPENMP
        omp_set_dynamic(0);
        int iel;
#pragma omp parallel shared(ndu, elein, elrho, elv, elq, min_cfl), firstprivate(GAMMA, nel), private(iel), num_threads(2)
        {
#pragma omp for reduction(min \
                          : min_cfl) schedule(static)
            for (iel = 0; iel < nel; iel++)
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

        auto dtLocal = std::min({0.25 - dt, dt * 1.02, 0.5 * min_cfl});

        // XXX --- MPI comms needed here to get global min. timestep ---

        error = MPI_Barrier(MPI_COMM_WORLD);
        error = MPI_Allreduce((const void *)&dtLocal, &dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

        // Predict half-step geometry and calculate pressure.

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
            else
            {
                elpSend = elp[nel - 1];
                elqSend = elq[nel - 1];
                error = MPI_Barrier(MPI_COMM_WORLD);
                for (auto i = 1; i < size; i++)
                {
                    if (rank == i - 1)
                    {
                        error = MPI_Send(&elpSend, 1, MPI_DOUBLE, i, tag_send, MPI_COMM_WORLD);
                        error = MPI_Send(&elqSend, 1, MPI_DOUBLE, i, tag_send, MPI_COMM_WORLD);
                    }
                    else if (rank == i)
                    {
                        error = MPI_Recv(&elpGhost, 1, MPI_DOUBLE, i - 1, tag_recv, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        error = MPI_Recv(&elqGhost, 1, MPI_DOUBLE, i - 1, tag_recv, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }
                f += elpGhost + elqGhost;
            }

            if (ielr < nel)
                f -= elp[ielr] + elq[ielr];
            else
            {
                elpSend = elp[0];
                elqSend = elq[0];
                error = MPI_Barrier(MPI_COMM_WORLD);
                for (auto i = 0; i < size - 1; i++)
                {
                    if (rank == i + 1)
                    {
                        error = MPI_Send(&elpSend, 1, MPI_DOUBLE, i, tag_send, MPI_COMM_WORLD);
                        error = MPI_Send(&elqSend, 1, MPI_DOUBLE, i, tag_send, MPI_COMM_WORLD);
                    }
                    else if (rank == i)
                    {
                        error = MPI_Recv(&elpGhost, 1, MPI_DOUBLE, i + 1, tag_recv, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        error = MPI_Recv(&elqGhost, 1, MPI_DOUBLE, i + 1, tag_recv, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }
                f -= elp[ielr] + elq[ielr];
            }

            // Calculate a=F/m and apply zero-acceleration boundary conditions.
            double a = f / std::max(1.0e-40, ndm[ind]);
            if ((ind == 0 && rank == 0) || (ind == nnd - 1 && rank == size - 1))
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

    auto _ndx = ndx.release();
    if (rank != size - 1)
    {
        if (nnd)
        {
            double *p = new double[nnd - 1];

            std::copy(_ndx, _ndx + nnd - 1, p);

            std::swap(_ndx, p);

            delete[] p;
        }
    }

    MPI_Finalize();
    // std::clock_t c_end = std::clock();
    // auto t_end = std::chrono::high_resolution_clock::now();
    // std::cout << "Original code time usage." << std::endl;
    // std::cout << std::fixed << "CPU time used: "
    //           << 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC << " ms\n"
    //           << "Wall clock time passed: "
    //           << std::chrono::duration<double, std::milli>(t_end - t_start).count()
    //           << " ms\n";
    // XXX --- Uncomment this line to write density data to stdout. ---
    // XXX --- MPI comms needed here to gather data to root process. ---
    // plot(nel, ndx.get(), elrho.get());

    // delete[] origin._ndx;
    // delete[] origin._ndx05;
    // delete[] origin._ndm;
    // delete[] origin._ndu;
    // delete[] origin._ndubar;
    // delete[] origin._elrho;
    // delete[] origin._elp;
    // delete[] origin._elq;
    // delete[] origin._elein;
    // delete[] origin._elv;
    // delete[] origin._elm;
    return 0;
}
