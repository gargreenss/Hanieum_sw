import os
from roboflow import Roboflow
from dotenv import load_dotenv

load_dotenv()
rf = Roboflow(api_key="-----------")
project = rf.workspace("youseongbins-workspace").project("my-first-project-p96tw")
version = project.version(5)
dataset = version.download("yolov8")

print("받은 위치:", dataset.location)



