#include "./src/Common.h"
#include "./src/RANSAC.h"
#include "./src/Visualizer.h"
#include <opencv2/opencv.hpp>

int main()
{
	using namespace std;

	//Solve RANSAC
	Common::Vec2 input[] = { {-2, 0}, {0, 0.9}, {2, 2.0}, {3, 6.5}, {4, 2.9}, {5, 8.8}, {6, 3.95}, {8, 5.03}, {10, 5.97}, {12, 7.1}, {13, 1.2}, {14, 8.2}, {16, 8.5},{18, 10.1} };
	i32 length = sizeof(input) / sizeof(Common::Vec2);
	Common::Vec2 output;
	Common::Vec2 outputls;
	RANSAC::LinearRegressor* regressor = new RANSAC::RANSACRegressor();
	RANSAC::LinearRegressor* regressorls = new RANSAC::LeastSquareLinearRegressor();
	
	regressor->fit(input, length, &output);
	regressorls->fit(input, length, &outputls);


	cout << "RANSAC:K=" << output.x << endl;
	cout << "RANSAC:B=" << output.y << endl;
	cout << "LeastSquare:K=" << outputls.x << endl;
	cout << "LeastSquare:B=" << outputls.y << endl;

	//Visualization
	Visualizer::Plot* plot = new Visualizer::Plot();
	Visualizer::Plot* plotls = new Visualizer::Plot();
	Visualizer::PlotUtil* plotUtil = new Visualizer::PlotUtil();

	plotUtil->adaptiveCreatePlot(input,length,plot);
	plotUtil->scatter(input,length,plot);
	plotUtil->line(output,plot);
	plotUtil->addXAxis(plot);
	plotUtil->addYAxis(plot);
	plotUtil->addTitle("RANSAC Result",plot,150);

	plotUtil->adaptiveCreatePlot(input,length,plotls);
	plotUtil->scatter(input,length,plotls);
	plotUtil->line(outputls,plotls);
	plotUtil->addXAxis(plotls);
	plotUtil->addYAxis(plotls);
	plotUtil->addYAxis(plotls);
	plotUtil->addTitle("LeastSquare Result",plotls,200);
	
	plotUtil->show(plot);
	
	return 0;
}
