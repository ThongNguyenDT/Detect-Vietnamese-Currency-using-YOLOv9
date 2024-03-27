from roboflow import Roboflow

rf = Roboflow(api_key="m8k8Zs21EaUpsthRO9Nr")
project = rf.workspace("first-qkz37").project("vietnamese-currency-lgi9i")
version = project.version(4)

version.deploy(model_type="yolov9", model_path="train\\exp2")