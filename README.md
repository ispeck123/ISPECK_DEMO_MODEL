WEIGHT FILE LINK----
https://ispeck-my.sharepoint.com/:f:/g/personal/saikat_ispeck_co/EidofZE9P5hDjHjx096I7vkBaZy4rb43rjrcuv9TxNDkxA?e=76T9Xo

Model Work flow------
  
This repository implements an ensemble model combining two architectures: a heavy machinery detection model based on YOLOv6 and a person detection model utilizing a pretrained YOLOv5.

The heavy machinery detection model is custom-trained to identify machinery objects, while the person detection model leverages a pretrained network. Upon detecting a heavy machinery-like object, a custom tracker follows its movement and dynamically creates a virtual Region of Interest (ROI) around it.

Our additional logic is structured such that if a person is detected within this virtual ROI, an alert is triggered, ensuring real-time monitoring and safety compliance.all logical changes have been done at /yolov6/core/inferer.py

**Step 1:**
install the requirements.txt 


**Step 2:**
  Download the weights folder from the one drive link that mentioned above and keep it at your working directory.(custome weight file name 'N109_Updated.pt')


**Step 3:**
  change the configuration from the Config.py file. Add video file path at "RTSP_URL" .


**step 4:**
  run the script "run.py"
  (python run.py)

