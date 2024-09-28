# include "buff_predict/solver.h"

namespace Buff
{
  AllParamSolver::AllParamSolver()
  {
    // 添加范围限制
    // problem.SetParameterLowerBound(param, 0, 0.780);
    // problem.SetParameterUpperBound(param, 0, 1.045);

    // problem.SetParameterLowerBound(param, 1, 1.884);
    // problem.SetParameterUpperBound(param, 1, 2.000);

    // problem.SetParameterLowerBound(param, 2, -M_PI);
    // problem.SetParameterUpperBound(param, 2, M_PI);

    // 求解类型
    options.linear_solver_type = ceres::DENSE_QR;  
    // 不输出过程消息
    options.minimizer_progress_to_stdout = false;  
    // 最小梯度
    options.gradient_tolerance = 0.1;
  }
}