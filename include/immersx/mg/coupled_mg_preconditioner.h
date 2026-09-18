#pragma once

#include <deal.II/multigrid/multigrid.h>

template <typename VectorType>
class CoupledMgPreconditioner {

    public:
        CoupledMgPreconditioner(Multigrid<VectorType> &mg): mg(mg) {

        }

        void vmult(VectorType &my_dst,
              const VectorType &my_src) const {
                mg.defect[0] = my_src;
                mg.vcycle();
                my_dst = mg.solution[0];
              }


    Multigrid<VectorType> & mg;
};