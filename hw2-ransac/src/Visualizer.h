#pragma once
#include "Common.h"
#include <opencv2/opencv.hpp>

namespace Visualizer {
	namespace Constant{
		const f64 vcPaddingLeft = 0.1;
		const f64 vcPaddingRight = 0.1;
		const f64 vcPaddingTop = 0.2;
		const f64 vcPaddingBottom = 0.1;
		const f64 vcGraphWidth = 1024;
		const f64 vcGraphHeight = 768;
	}
	struct Plot{
		f64 xmin,xmax,ymin,ymax;
		cv::Mat* mat;
	};
	class PlotUtil {
	public:
		void show(Plot* plot);
		void adaptiveCreatePlot(Common::Vec2* points,i32 pointCount, Plot* plot);
		void scatter(Common::Vec2* points, i32 pointCount, Plot* plot);
		void createPlot(f64 xmin,f64 xmax,f64 ymin, f64 ymax,Plot* plot);
		void line(Common::Vec2 coef,Plot* plot);
		void addXAxis(Plot* plot);
		void addYAxis(Plot* plot);
		void addTitle(std::string title,Plot* plot,i32 offset);
	};
}