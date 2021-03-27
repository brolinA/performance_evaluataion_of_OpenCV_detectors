# Performance evaluation of detector - descriptor pair

 This performance evaluation is done as a part of Sensor Fusion Nano Degree course. 6 different descriptors and 5 different detectors were implemented and the different combinations have been evaluated to assess the best pair of detector and descriptor for the purpose of detecting the car in front of the ego car.

Detectors implemented:
1. Harris.
2. BRISK.
3. FAST.
4. ORB.
5. SIFT.
6. AKAZE.

Descriptors implemented:
1. BRIEF.
2. ORB.
3. FREAK.
4. SIFT.
5. AKAZE.


The use case of the project is **Time To Collision (TTC)** detection which requires that the detector - descriptor pair be as fast as possible. From the performance evaluation done on the different detector - descriptor pair, I can conclude that the best pair to be used for **Time To Collision (TTC)** detection are

1. FAST – BRIEF.
2. FAST – ORB.
3. ORB - BRIEF.

Detailed report can be found in [performance_report.pdf](report/performance_report.pdf) and the realted data & graphs can be found in [performance_report.ods](report/performance_report.ods).