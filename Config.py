import os

class Config:
    DB_HOST = os.environ.get('DB_HOST', '')
    DB_PORT =  int(os.environ.get('DB_PORT', ))
    DB_USER = os.environ.get('DB_USER', '')
    DB_PASSWORD = os.environ.get('DB_PASSWORD', '')
    DB_SCHEMA = os.environ.get('DB_SCHEMA', '')

    pipeline_id = int(os.environ.get("pipeline_id", ))
    RTSP_URL = os.environ.get("RTSP_URL", "")


    # Esentinel Table Configurations
    TABLE_CAM_MASTER = ""
    TABLE_CLASS_MASTER = ""
    TABLE_MODEL_MASTER = ""
    TABLE_OBJECT_INSIGHT = ""
    TABLE_PIPELINE_MASTER = ""
    TABLE_PIPELINE_CLASS_MAP = ""


    # Data View & Capture Settings 
    INSIGHTS_UPDATE_MODE_TIME_INTERVAL = int(os.environ.get("INSIGHTS_UPDATE_MODE_TIME_INTERVAL", 300)) # in seconds   ###5 min
    SAVE_IMAGE = str(os.environ.get("SAVE_IMAGE", "yes")).lower() in ("yes", "true", "t", "1")
    DIRECTORY_SAVE_IMAGE = os.environ.get("DIRECTORY_SAVE_IMAGE", "OUT")
    SHOW_IMAGE = str(os.environ.get("SHOW_IMAGE", "no")).lower() in ("yes", "true", "t", "1")
    SAVE_INSIGHT = str(os.environ.get("SAVE_INSIGHT", "yes")).lower() in ("yes", "true", "t", "1")

    
   
    SKIP_FRAMES = int(os.environ.get('SKIP_FRAMES', 10))
    MAX_FRAMES_MOVEMENT = int(os.environ.get('MAX_FRAMES_MOVEMENT', 60))
    MAX_DIST_MOVEMENT = int(os.environ.get('MAX_DIST_MOVEMENT', 100))
    CONF_THRESH = float(os.environ.get("CONF_THRESH", 0.5))
    IOU_THRESH = float(os.environ.get("IOU_THRESH", 0.5))
    PERSON_CONF_THRESH = float(os.environ.get("PERSON_CONF_THRESH", 0.2))
    UPPER_ROI = int(os.environ.get("UPPER_ROI", 4))
    LOWER_ROI = int(os.environ.get("LOWER_ROI", 4))
    LEFT_ROI = int(os.environ.get("LEFT_ROI", 4))
    RIGHT_ROI = int(os.environ.get("RIGHT_ROI", 4))
    

    # Model Weight And Class File Settings
    MODEL_WEIGHT = os.environ.get('MODEL_WEIGHT', os.path.join("weights", 'N109_Updated.pt'))
    MODEL_CLASS_YML = os.environ.get('MODEL_CLASS_YML', 'data/dataset_heavy_machinery.yml')
    MODEL_ALERT_CLASS_LIST = ['Proximity Alert']
    MODEL_ROI = os.environ.get("MODEL_ROI", False) # False
