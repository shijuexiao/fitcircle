#pragma once
/*****************************************************************//**
 * \file   fitcircle.h
 * \brief  卡尺找圆工具，亚像素精度
 *
 * \author WYJ
 * \date 2023-8-10
 *********************************************************************/
 /*
 使用示例：
 int main()
 {
	 cv::Mat src;
	 src = cv::imread(str);
	 Fvision::cvfunc::fitcircle::fitCircleParams cireparams;
	 cireparams.p1 = cv::Point2d(278, 259);//圆心
	 cireparams.radius = 174;//半径
	 cireparams.segNum = 30;//卡尺数量
	 cireparams.sigma = 1.0;//平滑
	 cireparams.halfheight = 32;//卡尺半高度
	 cireparams.halfwidth = 10;//卡尺半宽度
	 cireparams.edgethreshold = 5;//最小边缘幅度
	 cireparams.edge_type = 2;//边缘类型，参考measure_select说明
	 cireparams.edge_polarity = 1;
	 Fvision::cvfunc::fitcircle fitcir;
	 std::vector<Fvision::cvfunc::fitcircle::circleResult> outcirs;
	 Fvision::cvfunc::fitcircle::edgePointsRes ciredges;
	 auto start1 = std::chrono::steady_clock::now();
	 fitcir.findCircle(src, cireparams, outcirs, ciredges);
	 auto end1 = std::chrono::steady_clock::now();
	 double ellipse21 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
	 std::cout << "找圆耗时:" << ellipse21 << "ms" << std::endl;
	 std::cout << "圆心:" << outcirs[0].p1 << "半径:" << outcirs[0].radius << "得分:" << outcirs[0].score << std::endl;
	 fitcir.drawCirCalipers(src, cireparams);
	 for (size_t i = 0; i < ciredges.edgePoints.size(); i++)
	 {
		 cv::circle(src, ciredges.edgePoints[i], 2, cv::Scalar(255, 0, 0));
	 }
	 cv::circle(src, outcirs[0].p1, outcirs[0].radius, cv::Scalar(0, 255, 0));
 }
 */
#include<vector>
#include<opencv2/opencv.hpp>
#ifdef MYCVDLL
#define MYCVAPI __declspec(dllexport)
#else
#define MYCVAPI  __declspec(dllimport)
#endif
namespace Fvision {
	namespace cvfunc
	{
		class MYCVAPI fitcircle
		{
		public:
			fitcircle();
			~fitcircle();
			/**
			 * 线拟合输出结果.
			 */
			struct circleResult
			{
				cv::Point2d p1{};//圆心
				double radius;//半径
				double score;//得分
				std::vector<cv::Point2d> edgePoints;//点集(有效点)
				std::vector<cv::Point2d> invalidedgePoints;//点集（无效点）
			};
			/**
			 * 边缘点集和对应幅度值.
			 */
			struct edgePointsRes
			{
				std::vector<cv::Point2d> edgePoints;//点集
				std::vector<float> amplitude;//幅度
			};

			/**
			 * 卡尺拟合参数.
			 */
			struct fitCircleParams
			{
				cv::Point2d p1{};//圆心
				double radius;//半径
				int segNum{ 8 };//卡尺数量
				double sigma{ 1.0 };//平滑
				double halfheight{ 10.0 };//卡尺半高度
				double halfwidth{ 3.0 };//卡尺半宽度
				int edgethreshold{ 5 };//最小边缘幅度
				int edge_type{ 0 };//边缘类型，参考measure_select说明
				int edge_polarity{ 0 };//边缘极性，参考measure_transition说明
				double score = 0.7;//最小得分，分值=用于计算的边缘点数/卡尺数量
				int num_instances = 1;//找到的实例最大个数
				int max_num_iterations = -1;//执行RANSAC算法的最大迭代次数，默认不限制
				double distance_threshold = 3.5;////使用随机搜索 算法 （RANSAC） 以拟合几何形状。如果点到几何形状的距离<distance_threshold，认为该点符合预期
			};
			/**
			 * \functionName  findCircle
			 * \brief 卡尺找圆工具.
			 *
			 * \param src：输入图像
			 * \param circleparams：输入圆信息
			 * \param outcircle：输出圆结果
			 * \param edges：输出边缘点信息
			*/
			void findCircle(cv::Mat& src, fitCircleParams& circleparams, std::vector<circleResult>& outcircle, edgePointsRes& edges);
			/**
			 * \functionName  drawCirCalipers
			 * \brief 绘制圆卡尺工具.
			 *
			 * \param src：输入图像
			 * \param circleparams：输入圆信息
			*/
			void drawCirCalipers(cv::Mat& src, fitCircleParams& circleparams);
		private:
			class impl;
			std::unique_ptr<impl>impl_;
		};
	}
}