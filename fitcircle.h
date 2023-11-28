#pragma once
/*****************************************************************//**
 * \file   fitcircle.h
 * \brief  ������Բ���ߣ������ؾ���
 *
 * \author WYJ
 * \date 2023-8-10
 *********************************************************************/
 /*
 ʹ��ʾ����
 int main()
 {
	 cv::Mat src;
	 src = cv::imread(str);
	 Fvision::cvfunc::fitcircle::fitCircleParams cireparams;
	 cireparams.p1 = cv::Point2d(278, 259);//Բ��
	 cireparams.radius = 174;//�뾶
	 cireparams.segNum = 30;//��������
	 cireparams.sigma = 1.0;//ƽ��
	 cireparams.halfheight = 32;//���߰�߶�
	 cireparams.halfwidth = 10;//���߰���
	 cireparams.edgethreshold = 5;//��С��Ե����
	 cireparams.edge_type = 2;//��Ե���ͣ��ο�measure_select˵��
	 cireparams.edge_polarity = 1;
	 Fvision::cvfunc::fitcircle fitcir;
	 std::vector<Fvision::cvfunc::fitcircle::circleResult> outcirs;
	 Fvision::cvfunc::fitcircle::edgePointsRes ciredges;
	 auto start1 = std::chrono::steady_clock::now();
	 fitcir.findCircle(src, cireparams, outcirs, ciredges);
	 auto end1 = std::chrono::steady_clock::now();
	 double ellipse21 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
	 std::cout << "��Բ��ʱ:" << ellipse21 << "ms" << std::endl;
	 std::cout << "Բ��:" << outcirs[0].p1 << "�뾶:" << outcirs[0].radius << "�÷�:" << outcirs[0].score << std::endl;
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
			 * �����������.
			 */
			struct circleResult
			{
				cv::Point2d p1{};//Բ��
				double radius;//�뾶
				double score;//�÷�
				std::vector<cv::Point2d> edgePoints;//�㼯(��Ч��)
				std::vector<cv::Point2d> invalidedgePoints;//�㼯����Ч�㣩
			};
			/**
			 * ��Ե�㼯�Ͷ�Ӧ����ֵ.
			 */
			struct edgePointsRes
			{
				std::vector<cv::Point2d> edgePoints;//�㼯
				std::vector<float> amplitude;//����
			};

			/**
			 * ������ϲ���.
			 */
			struct fitCircleParams
			{
				cv::Point2d p1{};//Բ��
				double radius;//�뾶
				int segNum{ 8 };//��������
				double sigma{ 1.0 };//ƽ��
				double halfheight{ 10.0 };//���߰�߶�
				double halfwidth{ 3.0 };//���߰���
				int edgethreshold{ 5 };//��С��Ե����
				int edge_type{ 0 };//��Ե���ͣ��ο�measure_select˵��
				int edge_polarity{ 0 };//��Ե���ԣ��ο�measure_transition˵��
				double score = 0.7;//��С�÷֣���ֵ=���ڼ���ı�Ե����/��������
				int num_instances = 1;//�ҵ���ʵ��������
				int max_num_iterations = -1;//ִ��RANSAC�㷨��������������Ĭ�ϲ�����
				double distance_threshold = 3.5;////ʹ��������� �㷨 ��RANSAC�� ����ϼ�����״������㵽������״�ľ���<distance_threshold����Ϊ�õ����Ԥ��
			};
			/**
			 * \functionName  findCircle
			 * \brief ������Բ����.
			 *
			 * \param src������ͼ��
			 * \param circleparams������Բ��Ϣ
			 * \param outcircle�����Բ���
			 * \param edges�������Ե����Ϣ
			*/
			void findCircle(cv::Mat& src, fitCircleParams& circleparams, std::vector<circleResult>& outcircle, edgePointsRes& edges);
			/**
			 * \functionName  drawCirCalipers
			 * \brief ����Բ���߹���.
			 *
			 * \param src������ͼ��
			 * \param circleparams������Բ��Ϣ
			*/
			void drawCirCalipers(cv::Mat& src, fitCircleParams& circleparams);
		private:
			class impl;
			std::unique_ptr<impl>impl_;
		};
	}
}