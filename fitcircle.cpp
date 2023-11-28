#include "fitcircle.h"
namespace Fvision {
	namespace cvfunc
	{
		class fitcircle::impl
		{
		private:
			/**
			 * 边缘极性.
			 */
			enum class measure_transition
			{
				all = 0,//所有极性
				positive,//正极性，从黑到白
				negative//负极性，从白到黑
			};
			/**
			 * 边缘选择.
			 */
			enum class measure_select
			{
				all = 0,//所有边缘
				first,//第一个边缘
				last,//最后一个边缘
				best//最强幅度的边缘
			};
			/**
			 * 边缘点集.
			 */
			struct edgePoints
			{
				std::vector<cv::Point2d> edgePoints_p;//正极性
				std::vector<float> amplitude_p;//正极性幅度
				std::vector<cv::Point2d> edgePoints_n;//负极性
				std::vector<float> amplitude_n;//负极性幅度
			};
		public:
			void findCircle(cv::Mat& src, fitCircleParams& circleparams, std::vector<circleResult>& outcircle, edgePointsRes& edges)
			{
				if (src.empty() || circleparams.radius < 0.5)
				{
					std::cout << "以上参数错误!" << std::endl;
					std::runtime_error err("以上参数错误!");
					return;
				}
				outcircle.clear();
				edges.amplitude.clear();
				edges.edgePoints.clear();
				cv::Mat gray, srctem;
				std::vector<cv::Point>cirpoints;
				cv::Point2f center = cv::Point2f(circleparams.p1.x, circleparams.p1.y);
				float r = circleparams.radius + circleparams.halfheight + 2;
				cv::Point tl = cv::Point(circleparams.p1.x - r, circleparams.p1.y - r);
				cv::Point bl = cv::Point(circleparams.p1.x + r, circleparams.p1.y + r);
				cv::Rect rect1 = cv::Rect(tl, bl);
				cropImg(src, srctem, rect1);
				if (srctem.channels() > 1)
				{
					cv::cvtColor(srctem, gray, cv::COLOR_BGR2GRAY);
				}
				else
				{
					gray = srctem;
				}

				cv::Point2f Roi[3];//转换后
				cv::Point2f orinRoi[3];//原始区域
				orinRoi[0] = rect1.tl();
				orinRoi[1] = cv::Point2f(rect1.tl().x + rect1.width, rect1.tl().y);
				orinRoi[2] = rect1.br();
				Roi[0] = cv::Point2f(0, 0);
				Roi[1] = cv::Point2f(rect1.width, 0);
				Roi[2] = cv::Point2f(rect1.width, rect1.height);
				cv::Mat transPoints, transPointsInv;
				transPoints = cv::getAffineTransform(orinRoi, Roi);
				cv::invertAffineTransform(transPoints, transPointsInv);
				transPointsInv.at<double>(0, 2) += 0.5;
				transPointsInv.at<double>(1, 2) += 0.5;
				std::vector<cv::Point2d>circlepoints, circlepoints_tem;
				circlepoints.push_back(circleparams.p1);
				cv::transform(circlepoints, circlepoints_tem, transPoints);
				circleparams.p1 = circlepoints_tem[0];
				std::vector<cv::RotatedRect>rect2;
				createRect2(circleparams, rect2);
				//使用完后赋值回来
				circleparams.p1 = circlepoints[0];
				edgePoints edgestem;
				getEdgePoints(gray, circleparams, rect2, edgestem);
				if (edgestem.edgePoints_p.size() > 0)
				{
					cv::transform(edgestem.edgePoints_p, edgestem.edgePoints_p, transPointsInv);
				}
				if (edgestem.edgePoints_n.size() > 0)
				{
					cv::transform(edgestem.edgePoints_n, edgestem.edgePoints_n, transPointsInv);
				}
				std::vector<cv::Point2d> realPoints_p;
				std::vector<cv::Point2d> realPoints_n;
				std::vector<cv::Point2d> invalidrealPoints_p;
				std::vector<cv::Point2d> invalidrealPoints_n;
				RansacCircleFiler(edgestem.edgePoints_p, realPoints_p, invalidrealPoints_p, circleparams.segNum, circleparams.score, circleparams.max_num_iterations, circleparams.distance_threshold);
				RansacCircleFiler(edgestem.edgePoints_n, realPoints_n, invalidrealPoints_n, circleparams.segNum, circleparams.score, circleparams.max_num_iterations, circleparams.distance_threshold);
				edges.edgePoints.insert(edges.edgePoints.begin(), edgestem.edgePoints_p.begin(), edgestem.edgePoints_p.end());
				edges.edgePoints.insert(edges.edgePoints.end(), edgestem.edgePoints_n.begin(), edgestem.edgePoints_n.end());
				edges.amplitude.insert(edges.amplitude.begin(), edgestem.amplitude_p.begin(), edgestem.amplitude_p.end());
				edges.amplitude.insert(edges.amplitude.end(), edgestem.amplitude_n.begin(), edgestem.amplitude_n.end());
				measure_transition transition = static_cast<measure_transition>(circleparams.edge_polarity);
				switch (transition)
				{
				case measure_transition::all:
					getcircleResults(circleparams, realPoints_p, invalidrealPoints_p, outcircle);
					getcircleResults(circleparams, realPoints_n, invalidrealPoints_n, outcircle);
					break;
				case measure_transition::positive:
					getcircleResults(circleparams, realPoints_p, invalidrealPoints_p, outcircle);
					break;
				case measure_transition::negative:
					getcircleResults(circleparams, realPoints_n, invalidrealPoints_n, outcircle);
					break;
				default:
					break;
				}
				if (circleparams.num_instances == 1 && outcircle.size() > 1)
				{
					outcircle.resize(1);
				}
				return;
			}
			void drawCirCalipers(cv::Mat& src, fitCircleParams& circleparams)
			{
				cv::Mat dst;
				if (src.channels() < 3)
				{
					cv::cvtColor(src, dst, cv::COLOR_GRAY2BGR);
				}
				else
				{
					dst = src;
				}
				cv::Point2d center = circleparams.p1;
				double radius = circleparams.radius;
				int segnum = circleparams.segNum;
				int rectw = std::ceil(circleparams.halfwidth * 2);
				int recth = std::ceil(circleparams.halfheight * 2);
				double deltdeg = 360. / segnum;

				for (int i = 0; i < segnum; i++)
				{
					double radian = (deltdeg * i * CV_PI) / 180.;
					cv::Point2d endpoint(center.x + radius * cos(radian), center.y + radius * sin(radian));
					double angle = angle_lx(center, endpoint);
					//旋转矩形0°为Y正方向（垂直向上）
					cv::RotatedRect rect(endpoint, cv::Size(rectw, recth), angle + 90);
					cv::circle(dst, center, radius, cv::Scalar(0, 0, 255));
					//绘制卡尺
					cv::Point2f vertex[4];
					rect.points(vertex);
					for (int i = 0; i < 4; i++)
					{
						cv::line(dst, vertex[i], vertex[(i + 1) % 4], cv::Scalar(255, 100, 200), 1);
					}
					cv::Point2d endpoint1(endpoint.x + recth * cos(radian), endpoint.y + recth * sin(radian));
					cv::arrowedLine(dst, endpoint, endpoint1, cv::Scalar(0, 255, 0));
				}
			}
		private:
			double angle_lx(cv::Point2d p1, cv::Point2d p2)
			{
				if (p1 == p2)
				{
					//同一个点
					std::cout << "两个点相同!" << std::endl;
					std::runtime_error err("两个点相同!");
					return 0;
				}
				cv::Point2d vector = p2 - p1;
				if (vector.x == 0)
				{
					if (vector.y > 0)
					{
						return 90;
					}
					else
					{
						return -90;
					}
				}
				double angle = (acos(pow(vector.x, 2) / (vector.x * sqrt(pow(vector.x, 2) + pow(vector.y, 2))))) * (180 / CV_PI);
				if (p1.y > p2.y)
				{
					angle = -angle;
				}
				return  angle;
			}
			double getsubpix(std::vector<cv::Point2d>& points, float* amptem)
			{
				//抛物线顶点式：y=a(x-b)^2+c
				if (points.size() < 3)
				{
					return -1;
				}
				cv::Mat matrix = cv::Mat_<float>(3, 3);
				matrix.at<float>(0, 0) = cv::pow(points[0].x, 2);
				matrix.at<float>(1, 0) = points[0].x;
				matrix.at<float>(2, 0) = 1;
				matrix.at<float>(0, 1) = cv::pow(points[1].x, 2);
				matrix.at<float>(1, 1) = points[1].x;
				matrix.at<float>(2, 1) = 1;
				matrix.at<float>(0, 2) = cv::pow(points[2].x, 2);
				matrix.at<float>(1, 2) = points[2].x;
				matrix.at<float>(2, 2) = 1;
				cv::Mat invMatrix;
				cv::invert(matrix, invMatrix, 1);
				cv::Mat matrix_y = cv::Mat_<float>(1, 3);
				matrix_y.at<float>(0, 0) = points[0].y;
				matrix_y.at<float>(0, 1) = points[1].y;
				matrix_y.at<float>(0, 2) = points[2].y;
				cv::Mat result = cv::Mat_<float>(1, 3);
				result = matrix_y * invMatrix;
				double a = result.at<float>(0, 0);
				double b = -0.5 * result.at<float>(0, 1) / a;
				float c = result.at<float>(0, 2) - a * b * b;
				*amptem = c;
				return b;
			}
			void calsubpix(cv::Mat& sobelimg1, std::vector<double>& indexs, std::vector<float>& values)
			{
				int num = indexs.size();
				if (num < 1)
				{
					return;
				}
				for (int i = 0; i < num; i++)
				{
					std::vector<cv::Point2d> points;
					double sobelindex = indexs[i];
					float val1 = sobelimg1.at<float>(0, sobelindex - 1);
					float val2 = sobelimg1.at<float>(0, sobelindex);
					float val3 = sobelimg1.at<float>(0, sobelindex + 1);
					points.emplace_back(cv::Point2d(sobelindex - 1., val1));
					points.emplace_back(cv::Point2d(sobelindex, val2));
					points.emplace_back(cv::Point2d(sobelindex + 1., val3));
					float tem;
					if (val1 == val2 && val1 == val3)
					{
						indexs[i] = sobelindex;
						values[i] = val1;
					}
					else
					{
						indexs[i] = getsubpix(points, &tem);
						values[i] = tem;
					}
				}
			}
			void createRect2(fitCircleParams& circleparams, std::vector<cv::RotatedRect>& rect2)
			{
				cv::Point2d center = circleparams.p1;
				double radius = circleparams.radius;
				int segnum = circleparams.segNum;
				int rectw = std::ceil(circleparams.halfwidth * 2);
				int recth = std::ceil(circleparams.halfheight * 2);
				double deltdeg = 360. / segnum;

				for (int i = 0; i < segnum; i++)
				{
					double radian = (deltdeg * i * CV_PI) / 180;
					cv::Point2d endpoint(center.x + radius * cos(radian), center.y + radius * sin(radian));
					double angle = angle_lx(center, endpoint);
					//旋转矩形0°为Y正方向（垂直向上）
					cv::RotatedRect rect(endpoint, cv::Size(rectw, recth), angle + 90);
					rect2.emplace_back(rect);
				}
			}
			void getEdgePoints(cv::Mat& gray, fitCircleParams& circleparams, std::vector<cv::RotatedRect>& rect2, edgePoints& edges)
			{
				measure_select select = static_cast<measure_select>(circleparams.edge_type);

				for (auto& temRect2 : rect2)
				{
					//映射后的矩形
					cv::Point2f transRoi[4];
					transRoi[0] = cv::Point2f(0, 0);
					transRoi[1] = cv::Point2f(circleparams.halfheight * 2, 0);
					transRoi[2] = cv::Point2f(circleparams.halfheight * 2, circleparams.halfwidth * 2);
					transRoi[3] = cv::Point2f(0, circleparams.halfwidth * 2);
					std::vector<float>values_p;//符合条件的正极性幅度值
					std::vector<double>indexs_p;//符合条件的正极性索引
					std::vector<float>values_n;//符合条件的负极性幅度值
					std::vector<double>indexs_n;//符合条件的负极性索引
						//仿射矩形映射到指定矩形
					cv::Point2f vertex[4];
					temRect2.points(vertex);
					cv::Mat transMatrix = cv::getPerspectiveTransform(vertex, transRoi);
					cv::Mat invMatrix = cv::getAffineTransform(transRoi, vertex);
					cv::Mat temimg;
					cv::warpPerspective(gray, temimg, transMatrix, cv::Size(circleparams.halfheight * 2, circleparams.halfwidth * 2), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
					//垂直投影
					int rows = temimg.rows;
					int cols = temimg.cols;
					cv::Mat img = cv::Mat::zeros(cv::Size(cols, 1), temimg.type());
					cv::reduce(temimg, img, 0, cv::REDUCE_AVG);
					//sigma和ksize关系公式sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
					int ksize = 1;//核大小
					ksize = (circleparams.sigma - 0.35) / 0.15;
					if (ksize % 2 == 0)
					{
						ksize -= 1;
					}
					if (ksize <= 0)
					{
						ksize = 1;
					}
					//对垂直投影图执行平滑操作
					cv::Mat gaussdst;
					cv::GaussianBlur(img, gaussdst, cv::Size(ksize, 1), circleparams.sigma);
					//对x方向求导
					cv::Mat sobelimg1, sobelimg2;
					//一阶核
					cv::Mat matrix = (cv::Mat_<float>(1, 3) << -0.5, 0, 0.5);
					cv::filter2D(gaussdst, sobelimg1, CV_32F, matrix);
					sobelimg1.setTo(0, cv::abs(sobelimg1) < circleparams.edgethreshold);
					cv::filter2D(sobelimg1, sobelimg2, CV_32F, matrix);
#pragma region 每个卡尺判断正负极性和最强边缘
					int cols1 = sobelimg2.cols;
					std::vector<float>temvalues_p;//符合条件的正极性幅度值
					std::vector<double>temindexs_p;//符合条件的正极性索引
					std::vector<float>temvalues_n;//符合条件的负极性幅度值
					std::vector<double>temindexs_n;//符合条件的负极性索引
					int maxindexs[2];
					int minindexs[2];
					cv::minMaxIdx(sobelimg1, 0, 0, minindexs, maxindexs);
					double maxindex = maxindexs[1]; //最大幅度值索引
					double minindex = minindexs[1]; //最大幅度值索引
					float maxvalue = sobelimg1.at<float>(0, maxindex);//最大幅度值
					float minvalue = sobelimg1.at<float>(0, minindex);//最大幅度值
					for (int i = 0; i < cols1 - 1; i++)
					{
						float x1 = sobelimg2.at<float>(0, i);
						float x2 = sobelimg2.at<float>(0, i + 1);
						float x13 = x1 * x2;
						float tem = sobelimg1.at<float>(0, i);
						float tem2 = sobelimg1.at<float>(0, i + 1);
						double index = i;
						if (x13 < 0)
						{
							if (abs(tem) < abs(tem2))
							{
								tem = tem2;
								index = i + 1;
							}
							if (x1 < 0)
							{
								//负极性边缘
								temvalues_n.emplace_back(tem);
								temindexs_n.emplace_back(index);
							}
							else
							{
								temvalues_p.emplace_back(tem);
								temindexs_p.emplace_back(index);
							}
						}
						else if (x1 == 0 && tem != 0 && i > 0 && i < cols1)
						{
							//拐点
							float x4 = sobelimg2.at<float>(0, i - 1);
							if (x4 * x2 < 0)
							{
								if (x4 < 0)
								{
									//负极性边缘
									temvalues_n.emplace_back(tem);
									temindexs_n.emplace_back(index);
								}
								else
								{
									temvalues_p.emplace_back(tem);
									temindexs_p.emplace_back(index);
								}
							}
						}
					}
					calsubpix(sobelimg1, temindexs_p, temvalues_p);
					calsubpix(sobelimg1, temindexs_n, temvalues_n);
					if (temindexs_p.size() > 0)
					{
						maxindex = *std::max_element(temindexs_p.begin(), temindexs_p.end());
						maxvalue = *std::max_element(temvalues_p.begin(), temvalues_p.end());
					}
					if (temindexs_n.size() > 0)
					{
						minindex = *std::min_element(temindexs_n.begin(), temindexs_n.end());
						minvalue = *std::min_element(temvalues_n.begin(), temvalues_n.end());
					}
#pragma endregion
#pragma region 极性跟边缘位置选择
					int num_p = temvalues_p.size();
					int num_n = temvalues_n.size();
					auto fucall = [&] {
						if (num_p)
						{
							values_p.insert(values_p.end(), temvalues_p.begin(), temvalues_p.end());
							indexs_p.insert(indexs_p.end(), temindexs_p.begin(), temindexs_p.end());
						}
						if (num_n)
						{
							values_n.insert(values_n.end(), temvalues_n.begin(), temvalues_n.end());
							indexs_n.insert(indexs_n.end(), temindexs_n.begin(), temindexs_n.end());
						}
					};

					auto fucfirst = [&] {
						if (num_p)
						{
							values_p.emplace_back(temvalues_p[0]);
							indexs_p.emplace_back(temindexs_p[0]);
						}
						if (num_n)
						{
							values_n.emplace_back(temvalues_n[0]);
							indexs_n.emplace_back(temindexs_n[0]);
						}
					};
					auto fuclast = [&] {
						if (num_p)
						{
							values_p.emplace_back(temvalues_p[temvalues_p.size() - 1]);
							indexs_p.emplace_back(temindexs_p[temindexs_p.size() - 1]);
						}
						if (num_n)
						{
							values_n.emplace_back(temvalues_n[temvalues_n.size() - 1]);
							indexs_n.emplace_back(temindexs_n[temindexs_n.size() - 1]);
						}
					};
					auto fucbest = [&] {
						if (maxvalue > 0)
						{
							values_p.emplace_back(maxvalue);
							indexs_p.emplace_back(maxindex);
						}
						if (minvalue < 0)
						{
							values_n.emplace_back(minvalue);
							indexs_n.emplace_back(minindex);
						}
					};
					switch (select)
					{
					case measure_select::all:
						fucall();
						break;
					case measure_select::first:
						fucfirst();
						break;
					case measure_select::last:
						fuclast();
						break;
					case measure_select::best:
						fucbest();
						break;
					default:
						break;
					}
					double y = std::floor(circleparams.halfwidth);//y坐标
					if (indexs_p.size() > 0)
					{
						std::vector<cv::Point2d>points_p;
						for (int i = 0; i < indexs_p.size(); i++)
						{
							cv::Point2d temP(indexs_p[i], y);
							points_p.emplace_back(temP);
						}
						std::vector<cv::Point2d>tempoints_p;
						cv::transform(points_p, tempoints_p, invMatrix);
						edges.edgePoints_p.insert(edges.edgePoints_p.end(), tempoints_p.begin(), tempoints_p.end());
					}
					if (indexs_n.size() > 0)
					{
						std::vector<cv::Point2d>points_n;
						for (int i = 0; i < indexs_n.size(); i++)
						{
							cv::Point2d temN(indexs_n[i], y);
							points_n.emplace_back(temN);
						}
						std::vector<cv::Point2d>tempoints_n;
						cv::transform(points_n, tempoints_n, invMatrix);
						edges.edgePoints_n.insert(edges.edgePoints_n.end(), tempoints_n.begin(), tempoints_n.end());
					}
					edges.amplitude_p.insert(edges.amplitude_p.end(), values_p.begin(), values_p.end());
					edges.amplitude_n.insert(edges.amplitude_n.end(), values_n.begin(), values_n.end());
#pragma endregion
				}
			}
			void getcircleResults(fitCircleParams& circleparams, std::vector<cv::Point2d>& points, std::vector<cv::Point2d>& invalidedgePoints, std::vector<circleResult>& result)
			{
				int num = points.size();
				if (num < 3)
				{
					return;
				}
				circleResult temResult;
				cv::Vec4f line_para;
				fitCenterByLeastSquares(points, temResult);
				double scr = 1. * num / circleparams.segNum;
				temResult.score = MIN(1., scr);
				temResult.edgePoints.insert(temResult.edgePoints.begin(), points.begin(), points.end());
				temResult.invalidedgePoints.insert(temResult.invalidedgePoints.begin(), invalidedgePoints.begin(), invalidedgePoints.end());
				result.push_back(temResult);

			}
			void RansacCircleFiler(const std::vector<cv::Point2d>& points, std::vector<cv::Point2d>& outPoints, std::vector<cv::Point2d>& invalidedgePoints, int segnum, double segscore, int max_num_iterations, double distance_threshold)
			{
				int n = points.size();
				if (n < 3)
				{
					return;
				}
				cv::RNG random;
				double bestScore = -1.;
				std::vector<cv::Point2d>vpdTemp;
				std::vector<cv::Point2d>vpdTempInvalid;
				int iterations;//迭代次数
				if (max_num_iterations != -1)
				{
					iterations = max_num_iterations;
				}
				else
				{
					iterations = log(1 - 0.99) / (log(1 - (1.00 / n))) * 10;
				}
				for (int k = 0; k < iterations; k++)
				{
					int i1 = 0, i2 = 0, i3 = 0;
					while (i1 == i2 || i1 == i3 || i2 == i3)
					{
						i1 = random(n);
						i2 = random(n);
						i3 = random(n);
					}
					cv::Point2d p1 = points[i1];
					cv::Point2d p2 = points[i2];
					cv::Point2d p3 = points[i3];
					if (checkline(p1, p2, p3))
					{
						//共线点 重新选择
						continue;
					}
					circleResult cir;
					getCircle(p1, p2, p3, cir);
					double score = 0;
					vpdTemp.clear();
					vpdTempInvalid.clear();
					for (int i = 0; i < n; i++)
					{
						double distance = fabs(sqrt(pow(points[i].x - cir.p1.x, 2) + pow(points[i].y - cir.p1.y, 2)) - cir.radius);
						if (distance < distance_threshold)
						{
							vpdTemp.push_back(points[i]);
							score += 1;
						}
						else
						{
							vpdTempInvalid.push_back(points[i]);
						}
					}

					if (score > bestScore)
					{
						bestScore = score;
						double scoreTemp = 1. * vpdTemp.size() / segnum;
						if (scoreTemp > segscore)
						{
							outPoints = vpdTemp;
							invalidedgePoints = vpdTempInvalid;
						}
						if (k >= iterations)
						{
							break;
						}
						if (max_num_iterations == -1)
						{
							//自适应迭代次数
							iterations = log(1 - 0.99) / (log(1 - (pow(scoreTemp, 2))));
						}
					}
				}
			}
			bool checkline(cv::Point2d& p1, cv::Point2d& p2, cv::Point2d& p3)
			{
				cv::Point2d tem = p2 - p1;
				cv::Point2d tem1 = p3 - p1;
				double d = tem.x * tem1.y - tem.y * tem1.x;
				if (d == 0)
				{
					return true;
				}
				return false;
			}
			void getCircle(cv::Point2d& p1, cv::Point2d& p2, cv::Point2d& p3, circleResult& cir)
			{
				double a = p1.x - p2.x;
				double b = p1.y - p2.y;
				double c = p1.x - p3.x;
				double d = p1.y - p3.y;
				double e = ((p1.x * p1.x - p2.x * p2.x) - (p2.y * p2.y - p1.y * p1.y)) / 2;
				double f = ((p1.x * p1.x - p3.x * p3.x) - (p3.y * p3.y - p1.y * p1.y)) / 2;

				// 圆心位置 
				double x = (e * d - b * f) / (a * d - b * c);
				double y = (a * f - e * c) / (a * d - b * c);
				double r = sqrt(pow(x - p1.x, 2) + pow(y - p1.y, 2));
				cir.p1.x = x;
				cir.p1.y = y;
				cir.radius = r;
			}
			void fitCenterByLeastSquares(std::vector<cv::Point2d> mapPoint, circleResult& cir)
			{
				double sumX = 0, sumY = 0;
				double sumXX = 0, sumYY = 0, sumXY = 0;
				double sumXXX = 0, sumXXY = 0, sumXYY = 0, sumYYY = 0;

				for (auto& p : mapPoint)
				{
					sumX += p.x;
					sumY += p.y;
					sumXX += pow(p.x, 2);
					sumYY += pow(p.y, 2);
					sumXY += p.x * p.y;
					sumXXX += pow(p.x, 3);
					sumXXY += pow(p.x, 2) * p.y;
					sumXYY += pow(p.y, 2) * p.x;
					sumYYY += pow(p.y, 3);
				}

				int pCount = mapPoint.size();
				double M1 = pCount * sumXY - sumX * sumY;
				double M2 = pCount * sumXX - sumX * sumX;
				double M3 = pCount * (sumXXX + sumXYY) - sumX * (sumXX + sumYY);
				double M4 = pCount * sumYY - sumY * sumY;
				double M5 = pCount * (sumYYY + sumXXY) - sumY * (sumXX + sumYY);

				double a = (M1 * M5 - M3 * M4) / (M2 * M4 - M1 * M1);
				double b = (M1 * M3 - M2 * M5) / (M2 * M4 - M1 * M1);
				double c = -(a * sumX + b * sumY + sumXX + sumYY) / pCount;

				//圆心XY 半径
				double xCenter = -0.5 * a;
				double yCenter = -0.5 * b;
				double radius = 0.5 * sqrt(a * a + b * b - 4 * c);
				cir.p1.x = xCenter;
				cir.p1.y = yCenter;
				cir.radius = radius;
			}
			bool isPointInRect(cv::Point P, cv::Rect rect) {
				cv::Point A = rect.tl();
				cv::Point B(rect.tl().x + rect.width, rect.tl().y);
				cv::Point C(rect.tl().x + rect.width, rect.tl().y + rect.height);
				cv::Point D(rect.tl().x, rect.tl().y + rect.height);
				int x = P.x;
				int y = P.y;
				int a = (B.x - A.x) * (y - A.y) - (B.y - A.y) * (x - A.x);
				int b = (C.x - B.x) * (y - B.y) - (C.y - B.y) * (x - B.x);
				int c = (D.x - C.x) * (y - C.y) - (D.y - C.y) * (x - C.x);
				int d = (A.x - D.x) * (y - D.y) - (A.y - D.y) * (x - D.x);
				if ((a >= 0 && b >= 0 && c >= 0 && d >= 0) || (a <= 0 && b <= 0 && c <= 0 && d <= 0)) {
					return true;
				}
				return false;
			}
			void cropImg(cv::Mat& srcImage, cv::Mat& dst, cv::Rect& rect)
			{
				cv::Mat destImage = cv::Mat::zeros(rect.height, rect.width, 0);// 目标图像  
												   // 获取可填充图像  
				int crop_x1 = cv::max(0, rect.x);
				int crop_y1 = cv::max(0, rect.y);
				int crop_x2 = cv::min(srcImage.cols - 1, rect.x + rect.width - 1); // 图像范围 0到cols-1, 0到rows-1  
				int crop_y2 = cv::min(srcImage.rows - 1, rect.y + rect.height - 1);
				int rows = srcImage.rows;
				int cols = srcImage.cols;
				cv::Rect imgRect(0, 0, cols, rows);
				if (!isPointInRect(rect.tl(), imgRect) && !isPointInRect(rect.br(), imgRect))
				{
					dst = destImage;
					return;
				}
				cv::Mat roiImage = srcImage(cv::Range(crop_y1, crop_y2 + 1), cv::Range(crop_x1, crop_x2 + 1));// 左包含，右不包含  
																										// 如果需要填边  
				int left_x = (-rect.x);
				int top_y = (-rect.y);
				int right_x = rect.x + rect.width - srcImage.cols;
				int down_y = rect.y + rect.height - srcImage.rows;

				if (top_y > 0 || down_y > 0 || left_x > 0 || right_x > 0)//只要存在边界越界的情况，就需要边界填充
				{
					left_x = (left_x > 0 ? left_x : 0);
					right_x = (right_x > 0 ? right_x : 0);
					top_y = (top_y > 0 ? top_y : 0);
					down_y = (down_y > 0 ? down_y : 0);
					cv::copyMakeBorder(roiImage, destImage, top_y, down_y, left_x, right_x, cv::BORDER_CONSTANT, cv::Scalar::all(0));//cv::Scalar(0,0,255)指定颜色填充
					// 自带填充边界函数，top_y, down_y, left_x, right_x为非负正数  
					// 而且I.cols = roi_img.cols + left_x + right_x, I.rows = roi_img.rows + top_y + down_y  
				}
				else//若不存在边界越界的情况，则不需要填充了
				{
					destImage = roiImage;
				}
				dst = destImage;
			}
		};
	}
}
namespace Fvision {
	namespace cvfunc
	{
		fitcircle::fitcircle() :impl_{ std::make_unique<impl>() }
		{
		}
		fitcircle::~fitcircle() = default;
		void fitcircle::findCircle(cv::Mat& src, fitCircleParams& circleparams, std::vector<circleResult>& outcircle, edgePointsRes& edges)
		{
			impl_->findCircle(src, circleparams, outcircle, edges);
		}
		void fitcircle::drawCirCalipers(cv::Mat& src, fitCircleParams& circleparams)
		{
			impl_->drawCirCalipers(src, circleparams);
		}
	}
}