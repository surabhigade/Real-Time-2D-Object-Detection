# 2DRealTimeObjectDetection

  ## Overview
  This project focuses on real-time 2D object detection, ensuring consistent identification of objects against a plain white background. The detection remains unaffected by camera translation, zoom, or rotation.
  ## Tasks involved
1. Task 1 involves separating objects from the background by converting pixel values to binary based on a specified threshold.

![image](https://github.com/user-attachments/assets/bda60730-45c2-44c5-925b-775f0409525c)


2. Task 2 involves applying morphological filtering techniques to refine thresholded images, removing noise and filling holes for improved object segmentation. Strategies such as erosion, dilation are employed to enhance image quality and recognition accuracy.
 
![image](https://github.com/user-attachments/assets/9b2f87a3-beac-434c-affe-8304a93c7108)

3. Task 3 involves running connected components analysis on the thresholded and cleaned image, identifying, and displaying distinct areas. The system disregards small regions, prioritizes larger, central regions, and may limit recognition to the largest N regions for efficiency.

![image](https://github.com/user-attachments/assets/36370e6f-5bb0-4f43-93b2-956d4a67f6ac)

4. Task 4 involves developing a function to compute features for each major region, including percent filled and bounding box height/width ratio, ensuring translation, scale, and rotation invariance. These features are essential for robust object analysis and recognition, providing valuable insights into the characteristics of identified regions.
 
![image](https://github.com/user-attachments/assets/fc3987c5-d90a-45df-87b1-8d8a045da325)

5. Task 5 involves implementing a training mode within the system to collect feature vectors from objects, assign labels, and store them in a database for subsequent use in classifying unknown objects, facilitating the creation of a labeled dataset essential for training the recognition system.

6. Task 6 involves enabling the system to classify new feature vectors using a known objects database and scaled Euclidean distance, labeling objects based on the closest match and indicating results on the output video stream, with an extension for handling unknown objects.
 
![image](https://github.com/user-attachments/assets/d0daf2f2-1bb4-4700-a20a-5a1fe8653d78)
