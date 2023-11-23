#include <stan/model/model_header.hpp>
#include <boost/math/interpolators/cardinal_cubic_b_spline.hpp>

namespace model_model_namespace {
  template<class T>
  using cspl = boost::math::interpolators::cardinal_cubic_b_spline<T>;
  
  template<typename T> std::vector<T> spl;
  template<typename T> std::vector<T> xmax;
  
  
  int setSpline(double& x, double& xm, std::vector<double>& y,
                std::ostream* pstream__ = nullptr) {
    spl<cspl<double> >.push_back(cspl<double>(y.begin(), y.end(), 0., x));
    xmax<double>.push_back(xm);
    return spl<cspl<double> >.size() - 1;
  }
  
  double getSVal(int idx, const double& x, std::ostream* pstream__ = nullptr) {
    if (x > xmax<double>[idx])
      return spl<cspl<double> >[idx](xmax<double>[idx]);
    return spl<cspl<double> >[idx](x);
  }
}
