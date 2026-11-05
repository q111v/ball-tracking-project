import json
from pathlib import Path

import cv2
import pytest

from balltrack.app import parser, run
from balltrack.evaluate import evaluate, iou
from balltrack.history import Trails
from tools.synthetic import generate


def test_trails_expire_and_do_not_bridge_gap():
    trails=Trails(length=3,max_gap=4)
    trails.update(0,[(1,(0,0))])
    trails.update(1,[(1,(1,0))])
    trails.update(3,[(1,(3,0))])
    assert trails.segments(1)==[((0,0),(1,0))]
    trails.update(8,[])
    assert not trails.points


def test_byte_tracker_occlusion(tmp_path):
    report=generate(tmp_path)
    assert report['metrics']['overall']['detection_success_rate_recall']==1
    assert report['metrics']['overall']['id_switches']==0
    assert report['same_id_after_occlusion']==1
    cap=cv2.VideoCapture(str(tmp_path/'synthetic.mp4'))
    assert cap.isOpened()
    count=0
    while cap.read()[0]:
        count+=1
    cap.release()
    assert count==120


def test_buffer_ablation(tmp_path):
    report=generate(tmp_path,buffer=2)
    assert report['same_id_after_occlusion']==0
    assert report['metrics']['overall']['id_switches']>=1


def test_evaluation_false_positive_and_switch():
    gt={'frames':[{'frame':0,'objects':[{'id':1,'bbox':[0,0,10,10]}]},
                  {'frame':1,'objects':[{'id':1,'bbox':[0,0,10,10]}]}]}
    pred={0:[{'id':5,'bbox':[0,0,10,10]},{'id':6,'bbox':[30,30,40,40]}],
          1:[{'id':7,'bbox':[0,0,10,10]}]}
    m=evaluate(gt,pred)['metrics']['overall']
    assert m['precision']==pytest.approx(2/3)
    assert m['id_switches']==1
    assert iou([0,0,1,1],[2,2,3,3])==0


def test_missing_input(tmp_path):
    args=parser().parse_args(['--video',str(tmp_path/'missing.mp4'),'--no-display'])
    with pytest.raises(FileNotFoundError):
        run(args)


def test_cli_exclusive():
    with pytest.raises(SystemExit):
        parser().parse_args(['--video','a.mp4','--camera','0'])


def test_empty_evaluation():
    assert evaluate({'frames':[]},{})['same_id_after_occlusion'] is None


def test_recovery_counts_failure_to_rematch():
    gt={'frames':[
        {'frame':0,'objects':[{'id':1,'bbox':[0,0,10,10]}]},
        {'frame':1,'objects':[{'id':1,'bbox':[0,0,10,10],'visible':False}]},
        {'frame':2,'objects':[{'id':1,'bbox':[0,0,10,10]}]}]}
    report=evaluate(gt,{0:[{'id':5,'bbox':[0,0,10,10]}]})
    assert report['occlusion_recoveries_evaluated']==1
    assert report['same_id_after_occlusion']==0
    assert report['occlusion_events_without_rematch']==1


def test_sparse_annotations_do_not_count_unannotated_predictions():
    report=evaluate({'frames':[{'frame':0,'objects':[]}]},
                    {8:[{'id':1,'bbox':[0,0,10,10]}]})
    assert report['metrics']['overall']['fp']==0
