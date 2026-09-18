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

class CoupledTransfer : public MGTransferBase< ImmersX::LA::MPI::BlockVector > {

    public:
        using Payload = TrilinosWrappers::internal::LinearOperatorImplementation::TrilinosPayload;

        CoupledTransfer() {}

        void restrict_and_add (const unsigned int from_level, ImmersX::LA::MPI::BlockVector &dst, const ImmersX::LA::MPI::BlockVector &src) const override;
        void prolongate_and_add (const unsigned int to_level, ImmersX::LA::MPI::BlockVector &dst, const ImmersX::LA::MPI::BlockVector &src) const override;
        void prolongate (const unsigned int to_level, ImmersX::LA::MPI::BlockVector &dst, const ImmersX::LA::MPI::BlockVector &src) const override;

        template <int dim, int spacedim>
        void build(const DoFHandler< dim, spacedim > &bg_dof_handler, 
          const DoFHandler< 1, spacedim > &fg_dof_handler);

    private:
        MGTransferPrebuilt<ImmersX::LA::MPI::Vector> bg_transfer;

};

//Se dichiaro i metodi in un file cc non funziona (?)

void
CoupledTransfer::restrict_and_add (const unsigned int from_level,
     ImmersX::LA::MPI::BlockVector &dst, const ImmersX::LA::MPI::BlockVector &src) const {
          
     bg_transfer.restrict_and_add(from_level, dst.block(0), src.block(0));
     dst.block(1) += src.block(1);
     dst.block(2) += src.block(2);
}

void
CoupledTransfer::prolongate_and_add (const unsigned int from_level,
     ImmersX::LA::MPI::BlockVector &dst, const ImmersX::LA::MPI::BlockVector &src) const {

     bg_transfer.prolongate_and_add(from_level, dst.block(0), src.block(0));
     dst.block(1) += src.block(1);
     dst.block(2) += src.block(2);
}

void
CoupledTransfer::prolongate (const unsigned int from_level,
     ImmersX::LA::MPI::BlockVector &dst, const ImmersX::LA::MPI::BlockVector &src) const {

     bg_transfer.prolongate(from_level, dst.block(0), src.block(0));
     dst.block(1) = src.block(1);
     dst.block(2) = src.block(2);
}

template <int dim, int spacedim>
void 
CoupledTransfer::build(const DoFHandler< dim, spacedim > &bg_dof_handler, 
          const DoFHandler< 1, spacedim > &fg_dof_handler) {

     bg_transfer.build(bg_dof_handler);

     //Per ora basta
}