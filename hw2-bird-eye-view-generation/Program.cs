using System;
using System.Collections.Generic;
using OpenCvSharp;
namespace CVHomework2_5_1950641
{
    class Program
    {
        static void Main(string[] args)
        {
            //Step1. Preparations
            int distortionOrder = 5;
            CameraCalibrationHelper ccHelper = new CameraCalibrationHelper();
            List<List<Point2f>> cornerList;
            List<List<Point3f>> chessboardList;
            Size imageSize;
            double[,] intrinsicMatrix;
            double[] distortionCoefficients = { 0, 0, 0, 0, 0 };
            Vec3d[] rotationVectors;
            Vec3d[] translationVectors;
            Size patternSize = new Size(8, 5);

            //Step2. Calibrate camera & Obtain Intrinsics
            //ccHelper.findCalibrationBoardsInFolder("C:\\Users\\huang\\Desktop\\Works\\CV\\Hw2_5\\images", patternSize, out cornerList, out chessboardList, out imageSize);
            //ccHelper.executeCalibration(chessboardList, cornerList, imageSize, distortionOrder, out intrinsicMatrix, out distortionCoefficients, out rotationVectors, out translationVectors);

           
            intrinsicMatrix = new double[3, 3];
            intrinsicMatrix[0, 0] = 496.5220;
            intrinsicMatrix[0, 1] = 0;
            intrinsicMatrix[0, 2] = 319.5182;
            intrinsicMatrix[1, 0] = 0;
            intrinsicMatrix[1, 1] = 495.4242;
            intrinsicMatrix[1, 2] = 240.7480;
            intrinsicMatrix[2, 0] = 0;
            intrinsicMatrix[2, 1] = 0;
            intrinsicMatrix[2, 2] = 1;

            distortionCoefficients[0] = 0.1815;
            distortionCoefficients[1] = -0.6388;
            distortionCoefficients[4] = 0.5354;

            //Print Intrinsic
            Console.WriteLine("=======Intrinsic=========");
            for(int i = 0; i < 3; i++)
            {
                for(int j = 0; j < 3; j++)
                {
                    Console.Write(intrinsicMatrix[i, j]+" ");
                }
                Console.Write("\n");
            }
            Console.WriteLine("=========================");
            Console.WriteLine("X Focal Length:" + intrinsicMatrix[0, 0]);
            Console.WriteLine("Y Focal Length:" + intrinsicMatrix[1, 1]);
            Console.WriteLine("X Translation:" + intrinsicMatrix[0, 2]);
            Console.WriteLine("Y Translation:" + intrinsicMatrix[1, 2]);
            Console.WriteLine("=======Distortion========");
            foreach (var i in distortionCoefficients)
            {
                Console.Write(i + " ");
            }
            Console.Write("\n");


            //Step3-5. Attach pattern & Determine CS
            var wcs = new List<Point3f>();
            for (int j = 0; j < patternSize.Height; j++)  
            {
                for (int i = 0; i < patternSize.Width; i++)
                {
                    wcs.Add(new Point3f(320 + (i-patternSize.Width/2) * 36, 240 + (j - patternSize.Height / 2) * 36 * 0.75f, 1));
                }
            }
            //Step4-5.Take image & Unidistort the target image
            Mat targetImage = Cv2.ImRead("C:\\Users\\huang\\Desktop\\Works\\CV\\Hw2_5\\images\\target1950641.jpg");
            Mat undistortedTargetImage;
            ccHelper.imageUndistortion(targetImage, intrinsicMatrix, distortionCoefficients, out undistortedTargetImage);


            //Step6. Find Pairs
            List<Point3f> ics;
            ccHelper.getPairs(undistortedTargetImage, patternSize, wcs, out ics);

            //Step7. Solve Homography
            Mat homoMat;
            ccHelper.homographMatrixSolver(ics, wcs, out homoMat);

            Console.WriteLine("=======Homograph WCS TO ICS=========");
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    Console.Write(homoMat.At<double>(i, j) + " ");
                }
                Console.Write("\n");
            }
            //Step7b. Warp Perspective
            Mat output;
            ccHelper.warpPerspective(undistortedTargetImage, homoMat, out output);
            Cv2.ImShow("Result", output);
            Cv2.WaitKey(0);
        }
    }
}
