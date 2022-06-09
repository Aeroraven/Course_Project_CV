using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using OpenCvSharp;
namespace CVHomework2_5_1950641
{
    class CameraCalibrationHelper
    {
        public void findCalibrationBoardSingle(Mat inImage, Size bSize,ref List<Point2f> corners, out bool flag)
        {
            var cornersProxy = OutputArray.Create(corners);
            bool found = Cv2.FindChessboardCorners(inImage, bSize, cornersProxy);
            Console.WriteLine("Image Size=" + inImage.Rows + "," + inImage.Cols);
            Console.WriteLine("Corner ArrLen = " + corners.Count);
            if (found)
            {
                Console.WriteLine("Found");
                //Cv2.DrawChessboardCorners(inImage, bSize, InputArray.Create(corners), found);
            }
            
            flag = found;
        }
        public void findCalibrationBoardsInFolder(string path, Size bSize, out List<List<Point2f>> cornerList, out List<List<Point3f>> chessboardPts, out Size imSize)
        {
            DirectoryInfo dInfo = new DirectoryInfo(path);
            FileInfo[] fInfo = dInfo.GetFiles();
            List<List<Point2f>> ret = new List<List<Point2f>>();
            List<List<Point3f>> cret = new List<List<Point3f>>();
            Size isz = new Size(0, 0);
            int a = 0;
            foreach(var file in fInfo)
            {
                a += 1;
                if (a > 50)
                {
                    break;
                }
                Console.WriteLine("Calibrating - " + file.FullName);
                var imageIn = Cv2.ImRead(file.FullName, ImreadModes.Grayscale);
                isz = imageIn.Size();
                if (isz.Width != 640)
                {
                    Console.WriteLine("Corrupted File");
                    continue;
                }
                var corners = new List<Point2f>();
                bool flag;
                findCalibrationBoardSingle(imageIn, bSize,ref corners,out flag);
                if (flag)
                {
                    var crets = new List<Point3f>();
                    for (int i = 0; i < bSize.Width; i++)
                    {
                        for (int j = 0; j < bSize.Height; j++)
                        {
                            crets.Add(new Point3f(i, j, 0));
                        }
                    }
                    cret.Add(crets);
                    ret.Add(corners);
                }
            }
            cornerList = ret;
            chessboardPts = cret;
            imSize = isz;
        }
        public void executeCalibration(List<List<Point3f>> objectPoints, List<List<Point2f>> imagePoints, Size imageSize, int distortionOrders,
            out double[,] intMat, out double[] distortion, out Vec3d[] rvecs, out Vec3d[] tvecs)
        {
            double[,] intrinsicMatrix = new double[3, 3];
            double[] distortionCoefficients = new double[distortionOrders];
            int nums = objectPoints.Count;
            Vec3d[] rotationVectors;
            Vec3d[] translationVectors;
            Size sSize = new Size(imageSize.Height, imageSize.Width);
            Cv2.CalibrateCamera(objectPoints, imagePoints, sSize, intrinsicMatrix, distortionCoefficients, out rotationVectors, out translationVectors);
            intMat = intrinsicMatrix;
            distortion = distortionCoefficients;
            rvecs = rotationVectors;
            tvecs = translationVectors;
        }
        public void imageUndistortion(Mat inImage,double[,] intrinsicMatrix, double[] distortionCoefficient, out Mat undistortedImage)
        {
            Mat output = new Mat(inImage.Size(),inImage.Type());
            OutputArray outputProxy = OutputArray.Create(output);
            Mat cameraMatrix = new Mat(new Size(3, 3), MatType.CV_64FC1);
            for(int i = 0; i < 3; i++)
            {
                for(int j = 0; j < 3; j++)
                {
                    cameraMatrix.At<double>(i, j) = intrinsicMatrix[i, j];
                }
            }
            InputArray distortionProxy = InputArray.Create(distortionCoefficient);
            Cv2.Undistort(inImage, outputProxy, cameraMatrix, distortionProxy);
            undistortedImage = output;
        }
        public void getPairs(Mat inImage,Size patternSize,List<Point3f> worldPoints,out List<Point3f> homoCorners)
        {
            //First Detect Points
            List<Point2f> nCorners = new List<Point2f>();
            List<Point3f> nhCorners = new List<Point3f>();
            bool flag;
            findCalibrationBoardSingle(inImage, patternSize, ref nCorners, out flag);
            for(int i =0;i<worldPoints.Count;i++)
            {
                var el = worldPoints[i];
                var es = nCorners[i];
                nhCorners.Add(new Point3f(nCorners[i].X, nCorners[i].Y, 1));
                Console.WriteLine("(" + el.X + "," + el.Y + "," + el.Z + ") <-> (" + es.X + "," + es.Y + ")");
            }
            homoCorners = nhCorners;
        }
        public void homographMatrixSolver(List<Point3f> srcPoints, List<Point3f> dstPoints, out Mat homographMatrix)
        {
            //OpenCV Homography
            List<Point2f> dstpi = new List<Point2f>();
            List<Point2f> srcpi = new List<Point2f>();
            for(int i = 0; i < srcPoints.Count; i++)
            {
                dstpi.Add(new Point2f(dstPoints[i].X, dstPoints[i].Y));
                srcpi.Add(new Point2f(srcPoints[i].X, srcPoints[i].Y));
            }

            InputArray srcp = InputArray.Create(srcpi);
            InputArray dstp = InputArray.Create(dstpi);
            homographMatrix = Cv2.FindHomography(srcp, dstp);
            return;
            //Create the coefficient matrix
            Mat coefMat = new Mat(new Size(9,srcPoints.Count * 2), MatType.CV_64FC1);
            for(int i = 0; i < srcPoints.Count; i++)
            {
                // 1
                coefMat.At<double>(2 * i, 0) = -srcPoints[i].X;
                coefMat.At<double>(2 * i, 1) = -srcPoints[i].Y;
                coefMat.At<double>(2 * i, 2) = -1;
                coefMat.At<double>(2 * i, 3) = 0;
                coefMat.At<double>(2 * i, 4) = 0;
                coefMat.At<double>(2 * i, 5) = 0;
                coefMat.At<double>(2 * i, 6) = dstPoints[i].X * srcPoints[i].X;
                coefMat.At<double>(2 * i, 7) = dstPoints[i].X * srcPoints[i].Y;
                coefMat.At<double>(2 * i, 8) = dstPoints[i].X * srcPoints[i].Z;
                // 2
                coefMat.At<double>(2 * i + 1, 0) = 0;
                coefMat.At<double>(2 * i + 1, 1) = 0;
                coefMat.At<double>(2 * i + 1, 2) = 0;
                coefMat.At<double>(2 * i + 1, 3) = -srcPoints[i].X;
                coefMat.At<double>(2 * i + 1, 4) = -srcPoints[i].Y;
                coefMat.At<double>(2 * i + 1, 5) = -1;
                coefMat.At<double>(2 * i + 1, 6) = dstPoints[i].Y * srcPoints[i].X;
                coefMat.At<double>(2 * i + 1, 7) = dstPoints[i].Y * srcPoints[i].Y;
                coefMat.At<double>(2 * i + 1, 8) = dstPoints[i].Y * srcPoints[i].Z;
            }
            Console.WriteLine("========Mat=========");
            for(int i = 0; i < srcPoints.Count * 2; i++)
            {
                for(int j = 0; j < 9; j++)
                {
                    Console.Write(coefMat.At<double>(i, j) + " ");
                }
                Console.Write("\n");
            }
            //Least Square Estimate
            double eigen;
            double[] eigenVec;
            Console.WriteLine("In");
            leastSquareApproxSolver(coefMat, out eigen, out eigenVec);

            Mat homoMat = new Mat(3, 3, MatType.CV_64FC1);
            for(int i = 0; i < 3; i++)
            {
                for(int j = 0; j < 3; j++)
                {
                    homoMat.At<double>(i, j) = eigenVec[i * 3 + j];
                }
            }
            homographMatrix = homoMat;
        }
        public void leastSquareApproxSolver(Mat inMat, out double optimalEigenValue, out double[] optimalEigenVector)
        {
            //Solve Ax=0, equivalent to minimize the eigenvalue of A'A
            Mat inMatTranspose = inMat.T();
            Mat clMat = inMatTranspose * inMat;
            Mat eigenVectors = new Mat(inMat.Size(), MatType.CV_64FC1);
            OutputArray eigenVectorsProxy = OutputArray.Create(eigenVectors);
            List<double> eigenValues = new List<double>();
            OutputArray eigenValuesProxy = OutputArray.Create<double>(eigenValues);

            //Calculate eigen
            Console.Write("Val1");
            Cv2.Eigen(clMat, eigenValuesProxy, eigenVectorsProxy);
            Console.Write("Val2");
            //Find the smallest eigenvalue
            int idx = 0;
            double ev = 1e50;
            Console.WriteLine("Cnt=" + eigenValues.Count);
            for(int i = 0; i < eigenValues.Count; i++)
            {
                if (ev > eigenValues[i])
                {
                    idx = i;
                    ev = eigenValues[i];
                }
            }

            //Return the best eigenvector
            double[] bestEigenVector = new double[inMat.Rows];
            for(int i = 0; i < inMat.Rows; i++)
            {
                bestEigenVector[i] = eigenVectors.Row(idx).At<double>(i);
            }
            optimalEigenValue = ev;
            optimalEigenVector = bestEigenVector;
        }
        public void warpPerspective(Mat inputImage,Mat matrix,out Mat transformedOutput)
        {
            Mat outputMat = new Mat(inputImage.Size(), inputImage.Type());
            OutputArray outputProxy = OutputArray.Create(outputMat);
            Cv2.WarpPerspective(inputImage, outputMat, matrix, inputImage.Size());
            transformedOutput = outputMat;
        }
       
    }
}
