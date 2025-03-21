WEIGHT FILE LINK----
https://ispeck-my.sharepoint.com/:f:/g/personal/saikat_ispeck_co/Eo23ds7xe3pJs8R2_5DP9zgBavhftiC4mhum5F6xj_wQcw?e=1VHhcF

Model Work flow------
  
This repository implements an ensemble model combining two architectures: a heavy machinery detection model based on YOLOv6 and a person detection model utilizing a pretrained YOLOv5.

The heavy machinery detection model is custom-trained to identify machinery objects, while the person detection model leverages a pretrained network. Upon detecting a heavy machinery-like object, a custom tracker follows its movement and dynamically creates a virtual Region of Interest (ROI) around it.

Our additional logic is structured such that if a person is detected within this virtual ROI, an alert is triggered, ensuring real-time monitoring and safety compliance
