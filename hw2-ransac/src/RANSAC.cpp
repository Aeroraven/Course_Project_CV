#include "RANSAC.h"

namespace RANSAC{
	void LeastSquareLinearRegressor::fit(Common::Vec2* ls, i32 listSize, Common::Vec2* res) {
		f64 sumX = 0, sumY = 0, avgX = 0, avgY = 0;
		f64 sumXY = 0, sumXX = 0;
		using namespace std;
		for (i32 i = 0; i < listSize; i++) {
			sumX += ls[i].x;
			sumY += ls[i].y;
			sumXY += ls[i].x * ls[i].y;
			sumXX += ls[i].x * ls[i].x;
		}
		avgX = sumX / (f64)listSize;
		avgY = sumY / (f64)listSize;
		res->x = ((f64)listSize * sumXY - sumX * sumY) / ((f64)listSize * sumXX - sumX * sumX);
		res->y = avgY - res->x * avgX;
		
		
	}
	RANSACRegressor::RANSACRegressor(i32 minimalSubsetSize, f64 inlierAcceptanceThreshold, f64 inlierCountThresholdRatio, i32 maxIterations) {
		this->minimalSubsetSize = minimalSubsetSize;
		this->inlierAcceptanceThreshold = inlierAcceptanceThreshold;
		this->inlierCountThresholdRatio = inlierCountThresholdRatio;
		this->maxIterations = maxIterations;
	}
	void RANSACRegressor::fit(Common::Vec2* ls, i32 listSize, Common::Vec2* res) {
		using namespace std;
		i32 countThresh = (i32)((f64)listSize * inlierCountThresholdRatio);
		i32* bestConsensusSet = new i32[listSize];
		i32* curInlierSet = new i32[listSize];
		i32 bestConsensusLength = 0;
		i32* randomIndices = new i32[minimalSubsetSize];
		f64 bestEstimLoss = 0;
		LinearRegressor* lsRegressor = new LeastSquareLinearRegressor();
		Common::Vec2* temp = new Common::Vec2[listSize];
		Common::Vec2 rtemp;
		for (i32 T = 0; T < maxIterations; T++) {
			cout << "Iteration " << T << endl;
			while (true) {
				i32 inliers = 0;
				//Find the smallest subset S
				for (i32 i = 0; i < minimalSubsetSize; i++) {
					while (true) {
						randomIndices[i] = rand() % listSize;
						i32 flag = true;
						for (i32 j = 0; j < i; j++) {
							if (randomIndices[j] == randomIndices[i]) {
								flag = false;
							}
						}
						
						temp[i].x = ls[randomIndices[i]].x;
						temp[i].y = ls[randomIndices[i]].y;
						if (flag) {
							break;
						}
					}
				}
				//Fit data
				lsRegressor->fit(temp, minimalSubsetSize, &rtemp);
				//Find inliers
				f64 mloss = 0.0;
				for (i32 i = 0; i < listSize; i++) {
					f64 dist = 0;
					dist = abs((rtemp.x) * ls[i].x - ls[i].y + rtemp.y) / sqrt(1 + rtemp.x * rtemp.x);
					if (dist < inlierAcceptanceThreshold) {
						curInlierSet[inliers++] = i;
						mloss += dist;
					}
					
				}
				//Check
				if (inliers < countThresh) {
					continue;
				}
				//Update the optimal set
				if (inliers > bestConsensusLength) {
					for (i32 i = 0; i < inliers; i++) {
						bestConsensusSet[i] = curInlierSet[i];
					}
					bestConsensusLength = inliers;
					bestEstimLoss = mloss;
				}
				else if(inliers == bestConsensusLength && mloss < bestEstimLoss) {
					for (i32 i = 0; i < inliers; i++) {
						bestConsensusSet[i] = curInlierSet[i];
					}
					bestConsensusLength = inliers;
					bestEstimLoss = mloss;
				}
				break;
			}
		}
		//Restimate using the optimal set
		for (i32 i = 0; i < bestConsensusLength; i++) {
			temp[i].x = ls[bestConsensusSet[i]].x;
			temp[i].y = ls[bestConsensusSet[i]].y;
		}
		lsRegressor->fit(temp, bestConsensusLength, res);
	}
}

