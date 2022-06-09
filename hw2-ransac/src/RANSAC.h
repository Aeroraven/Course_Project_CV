#pragma once


#include "Common.h"


namespace RANSAC {
	class LinearRegressor {
	public:
		virtual void fit(Common::Vec2* ls, i32 listSize,  Common::Vec2* res) {}
	};
	class LeastSquareLinearRegressor: public LinearRegressor {
		void fit(Common::Vec2* ls, i32 listSize, Common::Vec2* res) override;
	};
	class RANSACRegressor:public LinearRegressor {
	private:
		i32 minimalSubsetSize = 2;
		f64 inlierAcceptanceThreshold = 0.1;
		f64 inlierCountThresholdRatio = 0.6;
		i32 maxIterations = 100;
	public:
		RANSACRegressor(i32 minimalSubsetSize = 2, f64 inlierAcceptanceThreshold = 0.2, f64 inlierCountThresholdRatio = 0.6, i32 maxIterations = 100);
		void fit(Common::Vec2* ls, i32 listSize, Common::Vec2* res) override;
	};
}