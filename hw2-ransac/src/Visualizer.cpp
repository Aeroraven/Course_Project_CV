#include "./Visualizer.h"

#include <sstream>
namespace Visualizer{
    void PlotUtil::createPlot(f64 xmin,f64 xmax,f64 ymin, f64 ymax,Plot* plot){
        plot->xmax = xmax;
        plot->xmin = xmin;
        plot->ymax = ymax;
        plot->ymin = ymin;
        i32 type = CV_8UC3;
        plot->mat = new cv::Mat(Constant::vcGraphHeight, Constant::vcGraphWidth, type);
    }
    void PlotUtil::scatter(Common::Vec2* points, i32 pointCount, Plot* plot){
        for(i32 i=0;i<pointCount;i++){
            f64 xlPercent = (points[i].x - plot->xmin)/(plot->xmax-plot->xmin);
            i32 discreteXLeft = Constant::vcGraphWidth * Constant::vcPaddingLeft + xlPercent *(1.0-(Constant::vcPaddingRight+Constant::vcPaddingLeft)) * Constant::vcGraphWidth ;
            f64 ylPercent = (points[i].y - plot->ymin)/(plot->ymax-plot->ymin);
            i32 discreteYBt = Constant::vcGraphHeight * Constant::vcPaddingBottom + ylPercent * (1.0-(Constant::vcPaddingTop+Constant::vcPaddingBottom)) * Constant::vcGraphHeight;
            i32 discreteYBottom = Constant::vcGraphHeight - discreteYBt;
            cv::circle(*(plot->mat),cv::Point(discreteXLeft,discreteYBottom),6,cv::Scalar(255.0,0.0,0.0,255.0),-1);
        }
    }
    
    void PlotUtil::show(Plot* plot){
        cv::imshow("Press ESC to Close",*(plot->mat));
        cv::waitKey(0);
    }
    void PlotUtil::addTitle(std::string title,Plot* plot,i32 offset){
        cv::putText(*(plot->mat),title,cv::Point(Constant::vcGraphWidth/2.0-offset,Constant::vcPaddingTop / 2.0 * Constant::vcGraphHeight),1,2,cv::Scalar(0.0,0.0,0.0,255.0),2);
    }
    void PlotUtil::adaptiveCreatePlot(Common::Vec2* points,i32 pointCount,Plot* plot){
        f64 xmin = 1e40;
        f64 xmax = -1e40;
        f64 ymin = 1e40;
        f64 ymax = -1e40;
        for(i32 i=0;i<pointCount;i++){
            xmin = Min(xmin,points[i].x);
            ymin = Min(ymin,points[i].y);
            xmax = Max(xmax,points[i].x);
            ymax = Max(ymax,points[i].y);
        }
        f64 xspan = xmax-xmin;
        f64 yspan = ymax-ymin;
        xmin -= xspan * 0.08;
        ymin -= yspan * 0.08;
        xmax += xspan * 0.08;
        ymax += yspan * 0.08;
        createPlot(xmin,xmax,ymin,ymax,plot);
    }
    void PlotUtil::line(Common::Vec2 coef,Plot* plot){
        f64 xspan = (plot->xmax - plot->xmin);
        f64 yspan = (plot->ymax - plot->ymin);
        f64 xst = plot->xmin + xspan * 0.04;
        f64 xed = plot->xmax - xspan * 0.04;
        f64 yst = (coef.x * plot->xmin + coef.y) + yspan * 0.04;
        f64 yed = (coef.x * plot->xmax + coef.y) - yspan * 0.04;

        i32 xst1 = Constant::vcGraphWidth * Constant::vcPaddingLeft + ((xst - plot->xmin)/(plot->xmax-plot->xmin))*(1-Constant::vcPaddingLeft-Constant::vcPaddingRight)*Constant::vcGraphWidth;
        i32 yst1 = Constant::vcGraphHeight * Constant::vcPaddingBottom + ((yst - plot->ymin)/(plot->ymax-plot->ymin))*(1-Constant::vcPaddingTop-Constant::vcPaddingBottom)*Constant::vcGraphHeight;
        i32 xed1 = Constant::vcGraphWidth * Constant::vcPaddingLeft + ((xed - plot->xmin)/(plot->xmax-plot->xmin))*(1-Constant::vcPaddingLeft-Constant::vcPaddingRight)*Constant::vcGraphWidth;
        i32 yed1 = Constant::vcGraphHeight * Constant::vcPaddingBottom + ((yed - plot->ymin)/(plot->ymax-plot->ymin))*(1-Constant::vcPaddingTop-Constant::vcPaddingBottom)*Constant::vcGraphHeight;
        yst1 = Constant::vcGraphHeight - yst1;
        yed1 = Constant::vcGraphHeight - yed1;
        cv::line(*(plot->mat),cv::Point(xst1,yst1),cv::Point(xed1,yed1),cv::Scalar(0.0,0.0,255.0,255.0),2);
    }
    void PlotUtil::addXAxis(Plot* plot){
        i32 xst = Constant::vcGraphWidth * Constant::vcPaddingLeft;
        i32 yst = Constant::vcGraphHeight * (1-Constant::vcPaddingBottom);
        i32 xed = Constant::vcGraphWidth * (1-Constant::vcPaddingRight);
        cv::line(*(plot->mat),cv::Point(xst,yst),cv::Point(xed,yst),cv::Scalar(0.0,0.0,0.0,255.0),2);
        
        for(i32 i =0;i<=5;i++){
            std::stringstream oss;
            oss<<(plot->xmax-plot->xmin)*i/5+plot->xmin;
            cv::putText(*(plot->mat),oss.str(),cv::Point(xst+(xed-xst)/5*i-20,yst+30),1,1,cv::Scalar(0.0,0.0,0.0,255.0),1);
            cv::line(*(plot->mat),cv::Point(xst+(xed-xst)/5*i,yst),cv::Point(xst+(xed-xst)/5*i,yst+7),cv::Scalar(0.0,0.0,0.0,255.0),2);
        }
    }
    void PlotUtil::addYAxis(Plot* plot){
        i32 xst = Constant::vcGraphWidth * Constant::vcPaddingLeft;
        i32 yst = Constant::vcGraphHeight * (1-Constant::vcPaddingBottom);
        i32 yed = Constant::vcGraphHeight * Constant::vcPaddingTop;
        cv::line(*(plot->mat),cv::Point(xst,yst),cv::Point(xst,yed),cv::Scalar(0.0,0.0,0.0,255.0),2);
        
        for(i32 i =0;i<=5;i++){
            std::stringstream oss;
            oss<<std::setprecision(2)<<(plot->ymax-plot->ymin)*i/5+plot->ymin;
            cv::putText(*(plot->mat),oss.str(),cv::Point(xst-55,yst+(yed-yst)/5*i+5),1,1,cv::Scalar(0.0,0.0,0.0,255.0),1);
            cv::line(*(plot->mat),cv::Point(xst,yst+(yed-yst)/5*i),cv::Point(xst-7,yst+(yed-yst)/5*i),cv::Scalar(0.0,0.0,0.0,255.0),2);
        }
    }
}