#pragma once

#include <deal.II/multigrid/mg_base.h>
#include <deal.II/multigrid/mg_transfer.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_vector.h>

using namespace dealii;

class CoupledChebyshevSmoother {

    public:

    using Payload = TrilinosWrappers::internal::LinearOperatorImplementation::TrilinosPayload;
    using AdditionalData = TrilinosWrappers::PreconditionChebyshev::AdditionalData;
    
    CoupledChebyshevSmoother(): cg_shur_control(1000, 1e-12), cg_shur(cg_shur_control) {}

    void initialize(const ImmersX::LA::MPI::BlockSparseMatrix & block_matrix,
        const AdditionalData & smoother_data_bg) 
    {
        Ct =
            linear_operator<ImmersX::LA::MPI::Vector, ImmersX::LA::MPI::Vector, Payload>(block_matrix.block(0,2));
        C = transpose_operator<ImmersX::LA::MPI::Vector, ImmersX::LA::MPI::Vector, Payload>(Ct);

        /*
        std::cout << block_matrix.block(0,2).m() << " " << block_matrix.block(0,2).n() << "\n";
        std::cout << block_matrix.block(1,1).m() << " " << block_matrix.block(1,1).n() << "\n";
        std::cout << block_matrix.block(0,0).m() << " " << block_matrix.block(0,0).n() << "\n";
        std::cout << block_matrix.block(2,2).m() << " " << block_matrix.block(2,2).n() << "\n";
        std::cout << block_matrix.block(2,1).m() << " " << block_matrix.block(2,1).n() << "\n";
        */
        
        Mt = linear_operator<ImmersX::LA::MPI::Vector, ImmersX::LA::MPI::Vector, Payload>(block_matrix.block(1,2));
        M = transpose_operator<ImmersX::LA::MPI::Vector, ImmersX::LA::MPI::Vector, Payload>(Mt);

        this -> smoother_data_bg = smoother_data_bg;
        this -> smoother_data_fg = smoother_data_bg;

        cheby_prec_bg.initialize(block_matrix.block(0,0), smoother_data_bg);
        cheby_prec_fg.initialize(block_matrix.block(1,1), smoother_data_bg);

        cheby_op_bg = linear_operator<ImmersX::LA::MPI::Vector, ImmersX::LA::MPI::Vector, Payload>(cheby_prec_bg);
        cheby_op_fg = linear_operator<ImmersX::LA::MPI::Vector, ImmersX::LA::MPI::Vector, Payload>(cheby_prec_fg);

        approx_shur = -1 * C * cheby_op_bg * Ct + M * cheby_op_fg * Mt;
    }

    void vmult(ImmersX::LA::MPI::BlockVector &my_dst,
              const ImmersX::LA::MPI::BlockVector &my_src) const
    {
        my_dst = 0; 

        cg_shur.solve(approx_shur,
            my_dst.block(2),
            C*cheby_op_bg*my_src.block(0) - M * cheby_op_fg*my_src.block(1) - my_src.block(2),
            PreconditionIdentity()
        );

        my_dst.block(0) = cheby_op_bg * (my_src.block(0) - Ct *  my_dst.block(2));
        my_dst.block(1) = cheby_op_fg * (my_src.block(1) + Mt *  my_dst.block(2));
    }

    //Simmetrico
    void Tvmult(ImmersX::LA::MPI::BlockVector &my_dst,
              const ImmersX::LA::MPI::BlockVector &my_src) const
    {
        vmult(my_dst, my_src);
    }

    void clear() {
        
    }

    TrilinosWrappers::PreconditionChebyshev::AdditionalData smoother_data_bg;
    TrilinosWrappers::PreconditionChebyshev::AdditionalData smoother_data_fg;

    LinearOperator<ImmersX::LA::MPI::Vector, ImmersX::LA::MPI::Vector, Payload> cheby_op_bg;
    LinearOperator<ImmersX::LA::MPI::Vector, ImmersX::LA::MPI::Vector, Payload> cheby_op_fg;
    
    LinearOperator<ImmersX::LA::MPI::Vector, ImmersX::LA::MPI::Vector, Payload> approx_shur;

    LinearOperator<ImmersX::LA::MPI::Vector, ImmersX::LA::MPI::Vector, Payload> C;
    LinearOperator<ImmersX::LA::MPI::Vector, ImmersX::LA::MPI::Vector, Payload> Ct;

    LinearOperator<ImmersX::LA::MPI::Vector, ImmersX::LA::MPI::Vector, Payload> M;
    LinearOperator<ImmersX::LA::MPI::Vector, ImmersX::LA::MPI::Vector, Payload> Mt;

    SolverControl cg_shur_control;
    mutable SolverCG<ImmersX::LA::MPI::Vector> cg_shur;

    TrilinosWrappers::PreconditionChebyshev cheby_prec_bg;
    TrilinosWrappers::PreconditionChebyshev cheby_prec_fg;
};