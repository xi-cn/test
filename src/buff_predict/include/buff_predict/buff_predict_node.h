# ifndef BUFF_PREDICT_NODE
# define BUFF_PREDICT_NODE

# include <rclcpp/rclcpp.hpp>

namespace Buff
{
  class BuffPredictNode : public rclcpp::Node
  {
    BuffPredictNode() : Node("buff_predict_node")
    {
      
    }
  };
}


# endif