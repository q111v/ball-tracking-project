import cv2
import numpy as np
from ultralytics.engine.results import Results

from balltrack.color_assist import ColorAssist


def result(image, boxes):
    return Results(image,path='synthetic-test',names={32:'sports ball'},boxes=np.asarray(boxes,dtype=np.float32).reshape(-1,6))


def test_requires_yolo_seed_then_uses_visible_pixels_only():
    image=np.zeros((240,320,3),np.uint8)
    cv2.circle(image,(100,100),25,(0,230,0),-1)
    assist=ColorAssist(32)
    unseeded=result(image,[])
    assist.process(unseeded)
    assert len(unseeded.boxes)==0
    seeded=result(image,[[75,75,125,125,.9,32]])
    assist.process(seeded)
    moved=np.zeros_like(image)
    cv2.circle(moved,(110,105),25,(0,230,0),-1)
    missed=result(moved,[])
    assist.process(missed)
    assert len(missed.boxes)==1
    assert .1<float(missed.boxes.conf[0])<.25
    center=missed.boxes.xywh[0,:2].numpy()
    assert np.linalg.norm(center-[110,105])<3
    absent=result(np.zeros_like(image),[])
    assist.process(absent)
    assert len(absent.boxes)==0


def test_short_loss_resets_after_twelve_blank_frames():
    image=np.zeros((200,200,3),np.uint8)
    cv2.circle(image,(90,90),25,(0,230,0),-1)
    assist=ColorAssist(32)
    assist.process(result(image,[[65,65,115,115,.9,32]]))
    for _ in range(13):
        assist.process(result(np.zeros_like(image),[]))
    assert assist.hue is None
